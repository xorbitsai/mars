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

from typing import Any, Collection, List

import pandas as pd

from ..core import register_tileable_optimization_rule
from ...core import (
    OptimizationRecord,
    OptimizationRecordType,
    CommonGraphOptimizationRule,
)
from .....core.operand import Operand
from .....dataframe.core import (
    parse_index,
    DataFrameData,
    SeriesData,
    DataFrameGroupByData,
    SeriesGroupByData,
)
from .....dataframe.datasource.core import ColumnPruneSupportedDataSourceMixin
from .....dataframe.groupby.aggregation import DataFrameGroupByAgg
from .....dataframe.indexing.getitem import DataFrameIndex
from .....dataframe.merge import DataFrameMerge
from .....typing import OperandType, EntityType
from .....utils import implements
from .input_column_selector import InputColumnSelector


CAN_BE_OPTIMIZED_OP_TYPES = (DataFrameMerge, DataFrameGroupByAgg)


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
                if successor not in self.context:
                    selected_columns = selected_columns | set(successor.dtypes.index)
                else:
                    selected_columns = selected_columns | set(
                        list(self.context[successor][entity])
                    )
            return selected_columns
        else:
            if isinstance(entity, DataFrameData):
                return set(entity.dtypes.index)
            else:
                return {entity.name}

    def _select_columns(self):
        for entity in self._graph.topological_iter(reverse=True):
            if self._is_skipped_type(entity):
                continue

            cur_cols = set()
            successors = [
                successor
                for successor in self._graph.successors(entity)
                if successor in self.context and entity in self.context[successor]
            ]
            if successors:
                for successor in successors:
                    cur_cols = cur_cols | set(list(self.context[successor][entity]))
            else:
                if isinstance(entity, DataFrameData):
                    cur_cols = set(entity.dtypes.index)
                else:
                    cur_cols = {entity.name}

            self.context[entity] = InputColumnSelector.select_input_columns(
                entity, cur_cols
            )

    def _insert_getitem_nodes(self):
        pruned_nodes = []
        new_nodes = []
        datasource_nodes = []
        node_list = list(self._graph.topological_iter())
        for entity in node_list:
            if self._is_skipped_type(entity):
                continue

            selected_columns = self._get_selected_columns(entity)
            op = entity.op
            if isinstance(op, ColumnPruneSupportedDataSourceMixin):
                op.set_pruned_columns(list(selected_columns))
                self.effective = True
                pruned_nodes.append(entity)
                datasource_nodes.append(entity)
                continue

            if isinstance(op, CAN_BE_OPTIMIZED_OP_TYPES):
                predecessors = list(self._graph.predecessors(entity))
                for predecessor in predecessors:
                    if (
                        self._is_skipped_type(predecessor)
                        or predecessor in datasource_nodes
                    ):
                        continue

                    pruned_columns = list(self.context[entity][predecessor])

                    # new node init
                    new_node_op = DataFrameIndex(
                        col_names=pruned_columns,
                    )
                    new_params = predecessor.params.copy()
                    new_params["shape"] = (
                        new_params["shape"][0],
                        len(pruned_columns),
                    )
                    new_params["dtypes"] = new_params["dtypes"][pruned_columns]
                    new_params["columns_value"] = parse_index(
                        new_params["dtypes"].index, store_data=True
                    )
                    new_node = new_node_op.new_dataframe(
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
                    pruned_nodes.extend([predecessor])
                    new_nodes.append(new_node)
        return pruned_nodes, new_nodes

    def _update_tileable_params(
        self, pruned_nodes: List[EntityType], new_nodes: List[EntityType]
    ):
        # change dtypes and columns_value
        queue = [n for n in pruned_nodes]
        affected_nodes = set()
        while len(queue) > 0:
            node = queue.pop(0)
            nodes = self._graph.successors(node)
            for w in nodes:
                if w not in affected_nodes:
                    queue.append(w)
                    if (w not in new_nodes) and (not self._is_skipped_type(w)):
                        affected_nodes.add(w)

        for node in affected_nodes:
            selected_columns = self._get_selected_columns(node)
            if selected_columns:
                if isinstance(node, DataFrameData):
                    new_dtypes = pd.Series(
                        dict(
                            (col, dtype)
                            for col, dtype in node.dtypes.iteritems()
                            if col in selected_columns
                        )
                    )
                    new_columns_value = parse_index(new_dtypes.index, store_data=True)
                    node._dtypes = new_dtypes
                    node._columns_value = new_columns_value
                    node._shape = (node.shape[0], len(new_dtypes))

    @implements(CommonGraphOptimizationRule.apply)
    def apply(self, op: OperandType):
        self._select_columns()
        pruned_nodes, new_nodes = self._insert_getitem_nodes()
        self._update_tileable_params(pruned_nodes, new_nodes)

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
        return not isinstance(
            entity, (DataFrameData, SeriesData, DataFrameGroupByData, SeriesGroupByData)
        )
