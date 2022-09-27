import os
import tempfile

import pandas as pd
import pytest

from ...... import dataframe as md
from ......dataframe.arithmetic import DataFrameMul
from ......dataframe.core import DataFrameData, SeriesData
from ......dataframe.datasource.read_csv import DataFrameReadCSV
from ......dataframe.datasource.read_parquet import DataFrameReadParquet
from ......dataframe.groupby.aggregation import DataFrameGroupByAgg
from ......dataframe.indexing.getitem import DataFrameIndex
from ......dataframe.merge import DataFrameMerge
from ......optimization.logical.tileable import optimize


@pytest.fixture()
def gen_data1():
    with tempfile.TemporaryDirectory() as tempdir:
        df = pd.DataFrame(
            {
                "c1": [3, 4, 5, 3, 5, 4, 1, 2, 3],
                "c2": [1, 3, 4, 5, 6, 5, 4, 4, 4],
                "c3": list("aabaaddce"),
                "c4": list("abaaaddce"),
            }
        )

        df2 = pd.DataFrame(
            {
                "c1": [3, 3, 4, 5, 6, 5, 4, 4, 4],
                "c2": [1, 3, 4, 5, 6, 5, 4, 4, 4],
                "c3": list("aabaaddce"),
                "c4": list("abaaaddce"),
            }
        )
        file_path = os.path.join(tempdir, "test.csv")
        file_path2 = os.path.join(tempdir, "test2.csv")
        df.to_csv(file_path)
        df2.to_csv(file_path2)
        yield file_path, file_path2


@pytest.fixture()
def gen_data2():
    with tempfile.TemporaryDirectory() as tempdir:
        df = pd.DataFrame(
            {
                "c1": [3, 4, 5, 3, 5, 4, 1, 2, 3],
                "c2": [1, 3, 4, 5, 6, 5, 4, 4, 4],
                "c3": [1, 3, 4, 1, 1, 9, 4, 4, 4],
                "c4": [3, 0, 5, 3, 5, 4, 1, 2, 10],
            }
        )

        df2 = pd.DataFrame(
            {
                "cc1": [3, 4, 5, 3, 5, 4, 1, 2, 3],
                "cc2": [1, 6, 4, 5, 6, 5, 4, 4, 4],
                "cc3": [1, 3, 4, 1, 1, 9, 4, 8, 4],
                "cc4": [3, 0, 5, 3, 5, 4, 1, 2, 10],
            }
        )

        file_path = os.path.join(tempdir, "test.pq")
        file_path2 = os.path.join(tempdir, "test2.pq")
        df.to_parquet(file_path)
        df2.to_parquet(file_path2)
        yield file_path, file_path2


def test_group_by(setup, gen_data1):
    file_path, _ = gen_data1

    df1 = md.read_csv(file_path)
    c = df1.groupby("c1")["c2"].sum()

    graph = c.build_graph()
    optimize(graph)
    groupby_agg_node = graph.result_tileables[0]
    assert isinstance(groupby_agg_node, SeriesData)
    assert isinstance(groupby_agg_node.op, DataFrameGroupByAgg)
    assert type(groupby_agg_node.op) is DataFrameGroupByAgg
    assert groupby_agg_node.name == "c2"

    groupby_agg_node_preds = graph.predecessors(groupby_agg_node)
    assert len(groupby_agg_node_preds) == 1
    read_csv_node = groupby_agg_node_preds[0]
    assert isinstance(read_csv_node, DataFrameData)
    assert isinstance(read_csv_node.op, DataFrameReadCSV)
    assert len(read_csv_node.op.usecols) == 2
    assert len({"c1", "c2"} ^ set(read_csv_node.op.usecols)) == 0

    raw = pd.read_csv(file_path)
    pd_res = raw.groupby("c1")["c2"].sum()
    r = c.execute().fetch()
    pd.testing.assert_series_equal(r, pd_res)


def test_merge_on_one_column(setup, gen_data1):
    file_path, file_path2 = gen_data1
    df1 = md.read_csv(file_path)
    df2 = md.read_csv(file_path2)
    c = df1.merge(df2, left_on="c1", right_on="c1")["c1"]

    graph = c.build_graph()
    optimize(graph)

    index_node = graph.result_tileables[0]
    assert type(index_node.op) is DataFrameIndex

    index_node_preds = graph.predecessors(index_node)
    assert len(index_node_preds) == 1

    merge_node = index_node_preds[0]
    assert type(merge_node.op) is DataFrameMerge

    merge_node_preds = graph.predecessors(merge_node)
    assert len(merge_node_preds) == 2

    read_csv_node = merge_node_preds[0]
    read_csv_op = read_csv_node.op
    assert type(read_csv_op) is DataFrameReadCSV
    assert len(read_csv_op.usecols) == 1
    assert read_csv_op.usecols == ["c1"]

    r = c.execute().fetch()
    raw1 = pd.read_csv(file_path)
    raw2 = pd.read_csv(file_path2)
    expected = raw1.merge(raw2, left_on="c1", right_on="c1")["c1"]
    pd.testing.assert_series_equal(r, expected)


def test_merge_on_two_columns(setup, gen_data1):
    file_path, file_path2 = gen_data1
    df1 = md.read_csv(file_path)
    df2 = md.read_csv(file_path2)
    c = df1.merge(df2, left_on=["c1", "c2"], right_on=["c1", "c2"])[["c1", "c2"]]

    graph = c.build_graph()
    optimize(graph)

    index_node = graph.result_tileables[0]
    assert type(index_node.op) is DataFrameIndex
    assert len(index_node.op.col_names) == 2

    merge_node = graph.predecessors(index_node)[0]
    read_csv_node = graph.predecessors(merge_node)[0]
    assert type(read_csv_node.op) is DataFrameReadCSV

    use_cols = read_csv_node.op.usecols
    assert len(use_cols) == 2
    assert set(use_cols) & {"c1", "c2"} == {"c1", "c2"}
    assert len(set(use_cols) ^ {"c1", "c2"}) == 0

    r = c.execute().fetch()
    raw1 = pd.read_csv(file_path)
    raw2 = pd.read_csv(file_path2)
    expected = raw1.merge(raw2, left_on=["c1", "c2"], right_on=["c1", "c2"])[
        ["c1", "c2"]
    ]
    pd.testing.assert_frame_equal(r, expected)


def test_group_by_then_merge(setup, gen_data1):
    file_path, file_path2 = gen_data1
    df1 = md.read_csv(file_path)
    df2 = md.read_csv(file_path2)
    r_group_res = df1.groupby(["c1"])[["c2"]].sum()
    c = df2.merge(r_group_res, left_on=["c2"], right_on=["c2"])[["c1", "c3"]]
    graph = c.build_graph()
    optimize(graph)
    r = c.execute().fetch()

    raw1 = pd.read_csv(file_path)
    raw2 = pd.read_csv(file_path2)
    group_res = raw1.groupby(["c1"])[["c2"]].sum()
    expected = raw2.merge(group_res, left_on=["c2"], right_on=["c2"])[["c1", "c3"]]
    pd.testing.assert_frame_equal(r, expected)

    index_node = graph.result_tileables[0]
    assert type(index_node.op) is DataFrameIndex

    merge_node = graph.predecessors(index_node)[0]
    merge_node_preds = graph.predecessors(merge_node)

    df2_node = [n for n in merge_node_preds if type(n.op) is DataFrameReadCSV][0]
    assert set(df2_node.op.usecols) == {"c1", "c2", "c3"}

    df1_node = [
        n
        for n in graph._nodes
        if type(n.op) is DataFrameReadCSV and n.op.path == file_path
    ][0]
    assert type(df1_node.op) is DataFrameReadCSV
    assert set(df1_node.op.usecols) == {"c1", "c2"}


def test_merge_then_group_by(setup, gen_data2):

    file_path, file_path2 = gen_data2
    df1 = md.read_parquet(file_path)
    df2 = md.read_parquet(file_path2)

    c = (
        (
            ((df1 + 1) * 2).merge(df2, left_on=["c1", "c3"], right_on=["cc2", "cc4"])[
                ["c1", "cc4"]
            ]
            * 2
        )
        .groupby(["cc4"])
        .apply(lambda x: x / x.sum())
    )
    graph = c.build_graph()
    optimize(graph)
    r = c.execute().fetch()

    raw1 = pd.read_parquet(file_path)
    raw2 = pd.read_parquet(file_path2)
    expected = (
        (
            ((raw1 + 1) * 2).merge(raw2, left_on=["c1", "c3"], right_on=["cc2", "cc4"])[
                ["c1", "cc4"]
            ]
            * 2
        )
        .groupby(["cc4"])
        .apply(lambda x: x / x.sum())
    )
    pd.testing.assert_frame_equal(r, expected)

    read_parquet_nodes = [n for n in graph._nodes if type(n.op) is DataFrameReadParquet]
    assert len(read_parquet_nodes) == 2

    for n in read_parquet_nodes:
        assert len(n.op.get_columns()) == 2

    merge_node = [n for n in graph._nodes if type(n.op) is DataFrameMerge][0]
    merge_node_preds = graph.predecessors(merge_node)
    assert len(merge_node_preds) == 2

    inserted_node = [n for n in merge_node_preds if type(n.op) is DataFrameIndex][0]
    assert len(inserted_node.op.col_names) == 2
    assert set(inserted_node.op.col_names) == {"c1", "c3"}

    mul_node = graph.predecessors(inserted_node)[0]
    assert type(mul_node.op) is DataFrameMul
    assert set(mul_node.dtypes.index.tolist()) == {"c1", "c3"}


def test_two_merges(setup, gen_data2):
    file_path, file_path2 = gen_data2
    df1 = md.read_parquet(file_path)
    df2 = md.read_parquet(file_path2)
    c = (
        (df1 + 1)
        .merge((df2 + 2), left_on=["c2", "c3"], right_on=["cc1", "cc4"])[
            ["c2", "c4", "cc1", "cc2"]
        ]
        .merge(df2, left_on=["cc1"], right_on=["cc3"])
    )
    graph = c.build_graph()
    optimize(graph)
    r = c.execute().fetch()

    raw1 = pd.read_parquet(file_path)
    raw2 = pd.read_parquet(file_path2)

    expected = (
        (raw1 + 1)
        .merge((raw2 + 2), left_on=["c2", "c3"], right_on=["cc1", "cc4"])[
            ["c2", "c4", "cc1", "cc2"]
        ]
        .merge(raw2, left_on=["cc1"], right_on=["cc3"])
    )
    pd.testing.assert_frame_equal(r, expected)

    parquet_nodes = [n for n in graph._nodes if type(n.op) is DataFrameReadParquet]
    assert len(parquet_nodes) == 2

    # df1 read parquet push down
    df1_node = [n for n in parquet_nodes if n.op.path == file_path][0]
    assert set(df1_node.op.get_columns()) == {"c2", "c3", "c4"}

    # df2 read parquet not push down since it needs all the columns
    df2_node = [n for n in parquet_nodes if n.op.path == file_path2][0]
    assert df2_node.op.columns is None

    # prove that inserted nodes take effect
    inserted_nodes = [n for n in graph._nodes if type(n.op) is DataFrameIndex]
    assert len(inserted_nodes) == 3

    index_after_merge_node = [
        n for n in inserted_nodes if type(graph.predecessors(n)[0].op) is DataFrameMerge
    ][0]
    assert set(index_after_merge_node.op.col_names) == {"c2", "c4", "cc1", "cc2"}


def test_two_groupbys_with_multi_index(setup, gen_data2):
    file_path, _ = gen_data2
    df = md.read_parquet(file_path)
    c = (
        (df * 2)
        .groupby(["c2", "c3"])
        .apply(lambda x: x["c1"].sum() / x["c2"].mean())
        .reset_index()
        .groupby("c3")
        .agg(["min", "max"])
    )
    graph = c.build_graph()
    optimize(graph)
    r = c.execute().fetch()

    raw = pd.read_parquet(file_path)
    expected = (
        (raw * 2)
        .groupby(["c2", "c3"])
        .apply(lambda x: x["c1"].sum() / x["c2"].mean())
        .reset_index()
        .groupby("c3")
        .agg(["min", "max"])
    )
    pd.testing.assert_frame_equal(r, expected)

    apply_node = [n for n in graph._nodes if type(n.op) is DataFrameGroupByAgg][0]
    assert set(apply_node.columns.index_value._index_value._data) == {
        (0, "min"),
        (0, "max"),
        ("c2", "max"),
        ("c2", "min"),
    }

    # apply cannot push down
    read_parquet_node = [
        n
        for n in graph._nodes
        if type(n.op) is DataFrameReadParquet and n.op.path == file_path
    ][0]
    assert read_parquet_node.op.get_columns() is None
