# Copyright 2022 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Collection

import pandas as pd

from ..core import register_tileable_optimization_rule
from ...core import (
    OptimizationRecord,
    OptimizationRecordType,
    CommonGraphOptimizationRule,
)
from .....core.operand import Operand
from .....dataframe.core import parse_index, DataFrameData, SeriesData
from .....dataframe.datasource.core import ColumnPruneSupportedDataSourceMixin
from .....dataframe.groupby.aggregation import DataFrameGroupByAgg
from .....dataframe.indexing.loc import DataFrameLocGetItem
from .....dataframe.merge import DataFrameMerge
from .....typing import OperandType, EntityType
from .....utils import implements
from .input_column_selector import InputColumnSelector


@register_tileable_optimization_rule([Operand])
class ColumnPruningRule(CommonGraphOptimizationRule):
    context = {}

    def _get_selected_columns(self, entity: EntityType):
        selected_columns = set()
        successors = list(self._graph.successors(entity))
        if successors:
            for successor in successors:
                if self._is_skipped_type(successor):
                    continue

                selected_columns = selected_columns | set(
                    list(self.context[successor][entity])
                )
            return selected_columns
        else:
            if isinstance(entity, DataFrameData):
                return set(entity.dtypes.index)
            else:
                return {entity.name}

    @implements(CommonGraphOptimizationRule.apply)
    def apply(self, op: OperandType):
        for entity in self._graph.topological_iter(reverse=True):
            if self._is_skipped_type(entity):
                continue

            cur_cols = set()
            successors = self._graph.successors(entity)
            if successors:
                for successor in successors:
                    if successor in self.context and entity in self.context[successor]:
                        cur_cols = cur_cols | set(list(self.context[successor][entity]))
            else:
                if isinstance(entity, DataFrameData):
                    cur_cols = set(entity.dtypes.index)
                else:
                    cur_cols = {entity.name}

            self.context[entity] = InputColumnSelector.select_input_columns(entity, cur_cols)

        # Modify DAG
        node_list = list(self._graph.topological_iter())
        for entity in node_list:
            if self._is_skipped_type(entity):
                continue

            selected_columns = self._get_selected_columns(entity)
            # change dtypes and columns_value
            if selected_columns:
                if isinstance(entity, DataFrameData):
                    new_dtypes = pd.Series(
                        dict(
                            (col, dtype)
                            for col, dtype in entity.dtypes.iteritems()
                            if col in selected_columns
                        )
                    )
                    new_columns_value = parse_index(new_dtypes.index, store_data=True)
                    entity._dtypes = new_dtypes
                    entity._columns_value = new_columns_value
                    entity._shape = (entity._shape[0], len(new_dtypes))

            op = entity.op
            if isinstance(op, ColumnPruneSupportedDataSourceMixin):
                op.set_pruned_columns(list(selected_columns))
                self.effective = True

            if isinstance(op, DataFrameMerge) or isinstance(op, DataFrameGroupByAgg):
                predecessors = list(self._graph.predecessors(entity))
                for predecessor in predecessors:
                    if self._is_skipped_type(predecessor):
                        continue

                    new_node_indexes = list(self.context[entity][predecessor])

                    if not self._is_insert_node_needed(new_node_indexes, predecessor):
                        continue

                    # new node init
                    new_node_output_types = predecessor.op.output_types
                    new_node_op = DataFrameLocGetItem(
                        indexes=new_node_indexes,
                        output_types=new_node_output_types,
                    )
                    new_params = predecessor.params.copy()
                    new_params["shape"] = (
                        new_params["shape"][0],
                        len(new_node_indexes),
                    )
                    new_params["dtypes"] = new_params["dtypes"][new_node_indexes]
                    new_params["columns_value"] = parse_index(
                        new_params["dtypes"].index, store_data=True
                    )
                    new_node = new_node_op.new_tileable(
                        [predecessor], **new_params
                    ).data

                    # change edges and nodes
                    self._graph.remove_edge(predecessor, entity)
                    self._graph.add_node(new_node)

                    self._graph.add_edge(predecessor, new_node)
                    self._graph.add_edge(new_node, entity)

                    self._records.append_record(
                        OptimizationRecord(
                            predecessor, new_node, OptimizationRecordType.new
                        )
                    )
                    # update inputs
                    entity.inputs[entity.inputs.index(predecessor)] = new_node
                    self.effective = True

    def _is_insert_node_needed(
        self, target_cols: Collection[Any], entity: EntityType
    ) -> bool:
        """
        Whether to insert a new GetItem node
        Parameters
        ----------
        target_cols
        entity

        Returns
        -------

        """
        actual_set = set()
        for node in list(self._graph.successors(entity)):
            if self._is_skipped_type(node):
                continue
            actual_set = actual_set | set(self.context[node][entity])
        return len(actual_set ^ set(target_cols)) != 0

    @staticmethod
    def _is_skipped_type(entity: EntityType) -> bool:
        """
        If an entity is not a DataFrame or a Series, do not handle that.
        Parameters
        ----------
        entity

        Returns
        -------

        """
        return not (isinstance(entity, DataFrameData) or isinstance(entity, SeriesData))
