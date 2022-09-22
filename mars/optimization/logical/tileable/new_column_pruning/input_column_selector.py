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

from typing import Callable, Dict, List, Any

from mars.core import TileableData
from mars.dataframe.merge import DataFrameMerge


class InputColumnSelector:

    _OP_TO_SELECT_FUNCTION = {}

    @staticmethod
    def _select_all_input_columns(
        tileable_data: TileableData, required_cols: List[Any]
    ) -> Dict[TileableData, List[Any]]:
        ret = {}
        for _input in tileable_data.op.inputs:
            ret[_input] = _input.data.columns
        return ret

    @classmethod
    def register(
        cls,
        op_cls: str,
        func: Callable[[TileableData, List[Any]], Dict[TileableData, List[Any]]],
    ):
        cls._OP_TO_SELECT_FUNCTION[op_cls] = func

    @classmethod
    def select_input_columns(
        cls, tileable_data: TileableData, required_cols: List[Any]
    ) -> Dict[TileableData, List[Any]]:
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
        op_name = type(tileable_data.op).__name__
        if op_name in cls._OP_TO_SELECT_FUNCTION:
            return cls._OP_TO_SELECT_FUNCTION[op_name](tileable_data, required_cols)
        else:
            return cls._select_all_input_columns(tileable_data, required_cols)


def df_merge_operand_select_function(
    tileable_data: TileableData, required_cols: List[Any]
) -> Dict[TileableData, List[Any]]:
    op = tileable_data.op  # type: DataFrameMerge
    # TODO Only works for dataframe
    left_data = op.inputs[0]
    right_data = op.inputs[1]
    left_dtypes = left_data.dtypes
    right_dtypes = right_data.dtypes
    return {
        left_data: [col for col in left_dtypes.index.tolist() if col in required_cols],
        right_data: [
            col for col in right_dtypes.index.tolist() if col in required_cols
        ],
    }


InputColumnSelector.register("DataFrameMerge", df_merge_operand_select_function)
