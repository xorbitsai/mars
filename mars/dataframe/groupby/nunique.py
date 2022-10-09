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
from typing import List, Union

import numpy as np
import pandas as pd

from .aggregation import DataFrameGroupByAgg
from ...core import OutputType


class DataFrameGroupByAggNunique:
    @classmethod
    def _get_level_indexes(
        cls, op: "DataFrameGroupByAgg", data: pd.DataFrame
    ) -> List[int]:
        index_names = [data.index.name] if data.index.name else data.index.names
        indexes = np.array([i for i in range(len(index_names))])

        index_names = np.array(index_names)
        level = op.groupby_params["level"]
        if type(level) is int:
            indexes = [indexes[level]]
        elif type(level) is str:
            indexes = np.argwhere(np.isin(index_names, level)).ravel().tolist()
        else:
            level = list(level)
            if type(level[0]) is int:
                indexes = indexes[level].tolist()
            else:
                indexes = np.argwhere(np.isin(index_names, level)).ravel().tolist()
        return indexes

    @classmethod
    def _get_selection_columns(
        cls, op: "DataFrameGroupByAgg"
    ) -> Union[None, List[str]]:
        if "selection" in op.groupby_params:
            selection = op.groupby_params["selection"]
            if isinstance(selection, (tuple, list)):
                selection = [n for n in selection]
            else:
                selection = [selection]
            return selection
        return None

    @classmethod
    def get_execute_map_result(
        cls, op: "DataFrameGroupByAgg", in_data: pd.DataFrame
    ) -> Union[pd.DataFrame, pd.Series]:
        selections = cls._get_selection_columns(op)
        by_cols = op.raw_groupby_params["by"]
        if by_cols is not None:
            cols = (
                [*selections, *by_cols] if selections is not None else in_data.columns
            )
            res = in_data[cols].drop_duplicates(subset=cols).set_index(by_cols)
        else:  # group by level
            selections = selections if selections is not None else in_data.columns
            level_indexes = cls._get_level_indexes(op, in_data)
            in_data = in_data.reset_index()
            index_names = in_data.columns[level_indexes].tolist()
            cols = [*index_names, *selections]
            res = in_data[cols].drop_duplicates().set_index(index_names)

        if op.raw_groupby_params["sort"]:
            res = res.sort_index()

        if op.output_types[0] == OutputType.series:
            res = res.squeeze()

        return res

    @classmethod
    def get_execute_combine_result(
        cls, op: "DataFrameGroupByAgg", in_data: pd.DataFrame
    ) -> Union[pd.DataFrame, pd.Series]:
        # in_data.index.names means MultiIndex (groupby on multi cols)
        index_col = in_data.index.name or in_data.index.names
        res = in_data.reset_index().drop_duplicates().set_index(index_col)
        if op.output_types[0] == OutputType.series:
            res = res.squeeze()
        return res

    @classmethod
    def get_execute_agg_result(
        cls, op: "DataFrameGroupByAgg", in_data: pd.DataFrame
    ) -> Union[pd.DataFrame, pd.Series]:
        groupby_params = op.groupby_params.copy()
        cols = in_data.index.name or in_data.index.names
        by = op.raw_groupby_params["by"]

        if by is not None:
            if op.output_types[0] == OutputType.dataframe:
                groupby_params.pop("level", None)
                groupby_params["by"] = cols
                in_data = in_data.reset_index()
        else:
            groupby_params["level"] = op.raw_groupby_params["level"]

        res = in_data.groupby(**groupby_params).nunique()
        return res
