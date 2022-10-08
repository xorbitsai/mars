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
import logging

import numpy as np
import pandas as pd

from .aggregation import DataFrameGroupByAgg
from ..reduction.core import ReductionSteps
from ...core import OutputType
from ...serialization.serializables import DictField
from ...utils import lazy_import, estimate_pandas_size

cp = lazy_import("cupy", rename="cp")
cudf = lazy_import("cudf")

logger = logging.getLogger(__name__)


class DataFrameGroupByAggNunique(DataFrameGroupByAgg):
    # record all the origin groupby nunique parameters
    raw_groupby_params = DictField("raw_groupby_params")

    @classmethod
    def _get_func_infos(cls, op: "DataFrameGroupByAgg", df) -> ReductionSteps:
        return ReductionSteps(pre_funcs=[], agg_funcs=[], post_funcs=[])

    @classmethod
    def _get_level_indexes(cls, op: "DataFrameGroupByAggNunique", data: pd.DataFrame):
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
        cls, op: "DataFrameGroupByAggNunique", data: pd.DataFrame
    ):
        # TODO hanlde groupby params dict
        if "selection" in op.groupby_params:
            selection = op.groupby_params["selection"]
            if isinstance(selection, (tuple, list)):
                selection = [n for n in selection]
            else:
                selection = [selection]
            return selection
        return None

    @classmethod
    def _execute_map(cls, ctx, op: "DataFrameGroupByAggNunique"):
        xdf = cudf if op.gpu else pd
        in_data = ctx[op.inputs[0].key]
        if (
            isinstance(in_data, xdf.Series)
            and op.output_types[0] == OutputType.dataframe
        ):
            in_data = cls._series_to_df(in_data, op.gpu)

        selections = cls._get_selection_columns(op, in_data)
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

        if getattr(op, "size_recorder_name", None) is not None:
            # record_size
            raw_size = estimate_pandas_size(in_data)
            agg_size = estimate_pandas_size(res)
            size_recorder = ctx.get_remote_object(op.size_recorder_name)
            size_recorder.record(raw_size, agg_size)

        ctx[op.outputs[0].key] = (res,)

    @classmethod
    def _execute_combine(cls, ctx, op: "DataFrameGroupByAggNunique"):
        xdf = cudf if op.gpu else pd
        in_data = ctx[op.inputs[0].key][0]
        if (
            isinstance(in_data, xdf.Series)
            and op.output_types[0] == OutputType.dataframe
        ):
            in_data = cls._series_to_df(in_data, op.gpu)

        # in_data.index.names means MultiIndex (groupby on multi cols)
        index_col = in_data.index.name or in_data.index.names
        res = in_data.reset_index().drop_duplicates().set_index(index_col)
        if op.output_types[0] == OutputType.series:
            res = res.squeeze()
        ctx[op.outputs[0].key] = (res,)

    @classmethod
    def _execute_agg(cls, ctx, op: "DataFrameGroupByAggNunique"):
        xdf = cudf if op.gpu else pd
        out_chunk = op.outputs[0]

        in_data = ctx[op.inputs[0].key][0]
        if (
            isinstance(in_data, xdf.Series)
            and op.output_types[0] == OutputType.dataframe
        ):
            in_data = cls._series_to_df(in_data, op.gpu)

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
        ctx[out_chunk.key] = res
