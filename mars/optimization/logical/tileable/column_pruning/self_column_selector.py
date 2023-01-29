# Copyright 2022-2023 XProbe Inc.
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

from typing import Set, Any, Callable, Iterable

from mars.core import TileableData
from mars.dataframe.core import BaseDataFrameData, BaseSeriesData
from mars.dataframe.indexing.getitem import DataFrameIndex
from mars.dataframe.indexing.setitem import DataFrameSetitem
from mars.typing import OperandType


class SelfColumnSelector:

    _OP_TO_SELECT_FUNCTION = {}

    @classmethod
    def register(
        cls,
        op_cls: OperandType,
        func: Callable[[TileableData], Set[Any]],
    ) -> None:
        if op_cls not in cls._OP_TO_SELECT_FUNCTION:
            cls._OP_TO_SELECT_FUNCTION[op_cls] = func
        else:
            raise ValueError(f"key {op_cls} exists.")

    @classmethod
    def select(cls, tileable_data: TileableData) -> Set[Any]:
        """
        TODO: docstring
        """
        op_type = type(tileable_data.op)
        if op_type in cls._OP_TO_SELECT_FUNCTION:
            return cls._OP_TO_SELECT_FUNCTION[op_type](tileable_data)
        for op_cls in op_type.__mro__:
            if op_cls in cls._OP_TO_SELECT_FUNCTION:
                cls._OP_TO_SELECT_FUNCTION[op_type] = cls._OP_TO_SELECT_FUNCTION[op_cls]
                return cls._OP_TO_SELECT_FUNCTION[op_cls](tileable_data)
        return set()


def register_selector(op_type: OperandType) -> Callable:
    def wrap(selector_func: Callable):
        SelfColumnSelector.register(op_type, selector_func)
        return selector_func

    return wrap


@register_selector(DataFrameSetitem)
def df_setitem_select_function(tileable_data: TileableData):
    return {tileable_data.op.indexes}


@register_selector(DataFrameIndex)
def df_index_select_function(tileable_data: TileableData):
    if tileable_data.op.col_names:
        col_names = tileable_data.op.col_names
        if isinstance(col_names, Iterable):
            return set(tileable_data.op.col_names)
        else:
            return {tileable_data.op.col_names}
    else:
        if isinstance(tileable_data, BaseDataFrameData):
            return set(tileable_data.dtypes.index)
        elif isinstance(tileable_data, BaseSeriesData):
            return {tileable_data.name}


# TODO: handle other ops
