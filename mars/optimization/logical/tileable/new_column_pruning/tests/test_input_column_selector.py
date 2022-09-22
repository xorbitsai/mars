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

from typing import Dict, List, Any

from mars.core import EntityData
from ..input_column_selector import InputColumnSelector


class MockOperand:
    pass


class MockEntityData(EntityData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._op = MockOperand()


def test_register():
    def _select_input_columns(
        entity_data: EntityData, required_cols: List[Any]
    ) -> Dict[EntityData, List[Any]]:
        return {MockEntityData(): ["bar"]}

    InputColumnSelector.register("MockOperand", _select_input_columns)
    print(InputColumnSelector.select_input_columns(MockEntityData(), ["a", "b", "c"]))


def test_col_pruning(setup):
    import mars.dataframe as md

    left = md.DataFrame({"col1": (1, 2, 3), "col2": (4, 5, 6)})
    right = md.DataFrame({"col1": (1, 3), "col3": (5, 8)})
    joined = left.merge(right, left_on="col1", right_on="col1")
    input_columns = InputColumnSelector.select_input_columns(
        joined.data, ["col1", "col2"]
    )
    assert left.data in input_columns
    assert input_columns[left.data] == ["col1", "col2"]
    assert right.data in input_columns
    assert input_columns[right.data] == ["col1"]
