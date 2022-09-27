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

from typing import Dict, Any, Set

import pytest
from ......core import TileableData
from ......dataframe import DataFrame
from ..input_column_selector import InputColumnSelector


class MockOperand:
    pass


class MockEntityData(TileableData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._op = MockOperand()


def test_register():
    def _select_input_columns(
        tileable_data: TileableData, required_cols: Set[Any]
    ) -> Dict[TileableData, Set[Any]]:
        return {}

    InputColumnSelector.register(MockOperand, _select_input_columns)
    mock_data = MockEntityData()
    assert InputColumnSelector.select_input_columns(mock_data, {"foo"}) == {}

    # unregister
    with pytest.raises(AttributeError):
        InputColumnSelector.unregister(MockOperand)
        InputColumnSelector.select_input_columns(mock_data, {"foo"})


def test_col_pruning():
    left = DataFrame({"col1": (1, 2, 3), "col2": (4, 5, 6)})
    right = DataFrame({"col1": (1, 3), "col2": (4, 5), "col3": (5, 8)})
    joined = left.merge(right, left_on="col1", right_on="col3")
    input_columns = InputColumnSelector.select_input_columns(joined.data, {"col1"})
    assert left.data in input_columns
    assert input_columns[left.data] == {"col1"}
    assert right.data in input_columns
    assert input_columns[right.data] == {"col1", "col3"}


def test_df_group_by_agg():
    df: DataFrame = DataFrame(
        {
            "foo": (1, 1, 2, 2),
            "bar": (3, 4, 3, 4),
            "baz": (5, 6, 7, 8),
            "qux": (9, 10, 11, 12),
        }
    )

    s = df.groupby(by="foo")["baz"].sum()
    input_columns = InputColumnSelector.select_input_columns(s.data, {"baz"})
    assert len(input_columns) == 1
    assert df.data in input_columns
    assert input_columns[df.data] == {"foo", "baz"}

    s = df.groupby(by=["foo", "bar"]).sum()
    input_columns = InputColumnSelector.select_input_columns(s.data, {"baz"})
    assert len(input_columns) == 1
    assert df.data in input_columns
    assert input_columns[df.data] == {"foo", "bar", "baz"}

    s = df.groupby(by="foo").agg(["sum", "max"])
    input_columns = InputColumnSelector.select_input_columns(s.data, {"baz"})
    assert len(input_columns) == 1
    assert df.data in input_columns
    assert input_columns[df.data] == {"foo", "baz"}

    s = df.groupby(by="foo")["bar", "baz"].agg(["sum", "max"])
    input_columns = InputColumnSelector.select_input_columns(s.data, {"baz"})
    assert len(input_columns) == 1
    assert df.data in input_columns
    assert input_columns[df.data] == {"foo", "bar", "baz"}

    s = df.groupby(by="foo").agg(new_bar=("bar", "sum"), new_baz=("baz", "sum"))
    input_columns = InputColumnSelector.select_input_columns(s.data, {"new_bar"})
    assert len(input_columns) == 1
    assert df.data in input_columns
    assert input_columns[df.data] == {"foo", "bar", "baz"}


def test_df_merge(setup):
    left: DataFrame = DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6), 1: (7, 8, 9)})
    right = DataFrame({"foo": (1, 2), "bar": (4, 5), "baz": (5, 8), 1: (7, 8)})

    joined = left.merge(right, on=["foo"])

    input_columns = InputColumnSelector.select_input_columns(joined.data, {"foo"})
    assert left.data in input_columns
    assert input_columns[left.data] == {"foo"}
    assert right.data in input_columns
    assert input_columns[right.data] == {"foo"}

    input_columns = InputColumnSelector.select_input_columns(
        joined.data, {"foo", "baz"}
    )
    assert left.data in input_columns
    assert input_columns[left.data] == {"foo"}
    assert right.data in input_columns
    assert input_columns[right.data] == {"foo", "baz"}

    input_columns = InputColumnSelector.select_input_columns(
        joined.data, {"foo", "1_x"}
    )
    assert left.data in input_columns
    assert input_columns[left.data] == {"foo", 1}
    assert right.data in input_columns
    assert input_columns[right.data] == {"foo"}

    joined = left.merge(right, on=["foo", "bar"])
    input_columns = InputColumnSelector.select_input_columns(joined.data, {"baz"})
    assert left.data in input_columns
    assert input_columns[left.data] == {"foo", "bar"}
    assert right.data in input_columns
    assert input_columns[right.data] == {"foo", "bar", "baz"}

    joined = left.merge(right, on=["foo", "bar"])
    input_columns = InputColumnSelector.select_input_columns(
        joined.data, {"1_x", "1_y"}
    )
    assert left.data in input_columns
    assert input_columns[left.data] == {"foo", "bar", 1}
    assert right.data in input_columns
    assert input_columns[right.data] == {"foo", "bar", 1}


def test_arithmatic_ops(setup):
    df: DataFrame = DataFrame(
        {
            "foo": (1, 1, 2, 2),
            "bar": (3, 4, 3, 4),
            "baz": (5, 6, 7, 8),
            "qux": (9, 10, 11, 12),
        }
    )
    df = df


def test_select_all():
    df: DataFrame = DataFrame(
        {
            "foo": (1, 1, 2, 2),
            "bar": (3, 4, 3, 4),
            "baz": (5, 6, 7, 8),
            "qux": (9, 10, 11, 12),
        }
    )
    head = df.head()
    input_columns = InputColumnSelector.select_input_columns(head.data, {"foo"})
    assert len(input_columns) == 1
    assert head.data.inputs[0] in input_columns
    assert input_columns[head.data.inputs[0]] == {"foo", "bar", "baz", "qux"}
