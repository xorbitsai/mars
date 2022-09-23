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

from collections import defaultdict
from typing import Callable, Dict, Any, Set

from .....core import TileableData
from .....dataframe import NamedAgg
from .....dataframe.arithmetic.core import DataFrameBinOp, DataFrameUnaryOp
from .....dataframe.core import DataFrameData
from .....dataframe.groupby.aggregation import DataFrameGroupByAgg
from .....dataframe.indexing.getitem import DataFrameIndex
from .....dataframe.indexing.setitem import DataFrameSetitem
from .....dataframe.merge import DataFrameMerge
from .....typing import OperandType


class InputColumnSelector:

    _OP_TO_SELECT_FUNCTION = {}

    @staticmethod
    def select_all_input_columns(
        tileable_data: TileableData, required_cols: Set[Any]
    ) -> Dict[TileableData, Set[Any]]:
        ret = {}
        for inp in tileable_data.op.inputs:
            if isinstance(inp, DataFrameData):
                ret[inp] = required_cols.intersection(set(inp.dtypes.index))
            else:
                ret[inp] = required_cols.intersection({inp.name})
        return ret

    @staticmethod
    def select_required_input_columns(
        tileable_data: TileableData, required_cols: Set[Any]
    ) -> Dict[TileableData, Set[Any]]:
        ret = {}
        for inp in tileable_data.op.inputs:
            if isinstance(inp, DataFrameData):
                ret[inp] = required_cols.intersection(set(inp.dtypes.index))
            else:
                ret[inp] = required_cols.intersection({inp.name})
        return ret

    @classmethod
    def register(
        cls,
        op_cls: OperandType,
        func: Callable[[TileableData, Set[Any]], Dict[TileableData, Set[Any]]],
    ):
        cls._OP_TO_SELECT_FUNCTION[op_cls] = func

    @classmethod
    def unregister(cls, op_cls: OperandType):
        if op_cls in cls._OP_TO_SELECT_FUNCTION:
            del cls._OP_TO_SELECT_FUNCTION[op_cls]

    @classmethod
    def select_input_columns(
        cls, tileable_data: TileableData, required_cols: Set[Any]
    ) -> Dict[TileableData, Set[Any]]:
        """
        Get the column pruning results of given tileable data.

        If the argument 'additional' is passed, then it is appended after the main info.

        Parameters
        ----------
        tileable_data : TileableData
            The tileable data to be processed.
        required_cols: List[Any]
            Names of columns required by the successors of the given tileable data. The data type can be int or str.
        Returns
        -------
        Dict[TileableData: List[Any]]
            A dictionary that represents the column pruning results. For every key-value pairs in the dictionary, the
            key is a predecessor of the given tileable data, and the value is a list of column names that the given
            tileable data depends on.
        """
        op_type = type(tileable_data.op)
        if op_type in cls._OP_TO_SELECT_FUNCTION:
            return cls._OP_TO_SELECT_FUNCTION[op_type](tileable_data, required_cols)
        for op_cls in op_type.__mro__:
            if op_cls in cls._OP_TO_SELECT_FUNCTION:
                cls._OP_TO_SELECT_FUNCTION[op_type] = cls._OP_TO_SELECT_FUNCTION[op_cls]
                return cls._OP_TO_SELECT_FUNCTION[op_cls](tileable_data, required_cols)
        return cls.select_all_input_columns(tileable_data, required_cols)


def df_merge_select_function(
    tileable_data: TileableData, required_cols: Set[Any]
) -> Dict[TileableData, Set[Any]]:
    op: DataFrameMerge = tileable_data.op
    left_data: DataFrameData = op.inputs[0]
    right_data: DataFrameData = op.inputs[1]

    ret = defaultdict(set)
    for df, suffix in zip([left_data, right_data], op.suffixes):
        for col in df.dtypes.index:
            if col in required_cols:
                ret[df].add(col)
            else:
                suffix_col = str(col) + suffix
                if suffix_col in required_cols:
                    ret[df].add(col)

    if op.on is not None:
        if isinstance(op.on, (list, tuple)):
            ret[left_data].update(op.on)
            ret[right_data].update(op.on)
        else:
            ret[left_data].add(op.on)
            ret[right_data].add(op.on)

    if op.left_on is not None:
        if isinstance(op.left_on, (list, tuple)):
            ret[left_data].update(op.left_on)
        else:
            ret[left_data].add(op.left_on)

    if op.right_on is not None:
        if isinstance(op.right_on, (list, tuple)):
            ret[right_data].update(op.right_on)
        else:
            ret[right_data].add(op.right_on)

    return ret


def df_groupby_agg_select_function(
    tileable_data: TileableData, required_cols: Set[Any]
) -> Dict[TileableData, Set[Any]]:
    op: DataFrameGroupByAgg = tileable_data.op
    inp: DataFrameData = op.inputs[0]
    by = op.groupby_params["by"]
    selection = op.groupby_params.get("selection", None)
    raw_func = op.raw_func

    selected_cols = set()
    # group by keys should be included
    if isinstance(by, (list, tuple)):
        for col in by:
            if col in inp.dtypes.index:
                # exclude index
                selected_cols.update(by)
    else:
        if by in inp.dtypes.index:
            # exclude index
            selected_cols.add(by)

    # add agg columns
    if op.raw_func is not None:
        if isinstance(raw_func, dict):
            selected_cols.update(set(raw_func.keys()))
        else:
            # no specified agg columns
            # required_cols should always be a subset of selection
            selected_cols.update(required_cols)
            if selection is not None:
                selected_cols.update(set(selection)) if isinstance(
                    selection, (list, tuple)
                ) else selected_cols.add(selection)
    elif op.raw_func_kw:
        # add renamed columns
        for new_col, origin in op.raw_func_kw.items():
            if isinstance(origin, NamedAgg):
                selected_cols.add(origin.column)
            else:
                assert isinstance(origin, tuple)
                selected_cols.add(origin[0])

    return {inp: selected_cols}


InputColumnSelector.register(DataFrameMerge, df_merge_select_function)
InputColumnSelector.register(
    DataFrameSetitem, InputColumnSelector.select_required_input_columns
)
InputColumnSelector.register(
    DataFrameBinOp, InputColumnSelector.select_required_input_columns
)
InputColumnSelector.register(
    DataFrameUnaryOp, InputColumnSelector.select_required_input_columns
)
InputColumnSelector.register(
    DataFrameIndex, InputColumnSelector.select_required_input_columns
)
InputColumnSelector.register(DataFrameGroupByAgg, df_groupby_agg_select_function)
