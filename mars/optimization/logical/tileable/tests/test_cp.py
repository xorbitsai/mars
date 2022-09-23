import os
import shutil
import tempfile

import pandas as pd
import pytest

from .. import optimize
from ..... import dataframe as md
from .....dataframe.core import SeriesData, DataFrameData
from .....dataframe.datasource.read_csv import DataFrameReadCSV
from .....dataframe.groupby.aggregation import DataFrameGroupByAgg


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

    # clean
    try:
        shutil.rmtree(tempdir)
    except:
        shutil.rmtree(tempdir, ignore_errors=True)


def test_group_by(setup, gen_data1):
    file_path, _ = gen_data1

    df1 = md.read_csv(file_path)
    c = df1.groupby("c1")["c2"].sum()

    graph = c.build_graph()
    optimize(graph)
    groupby_agg_node = graph.result_tileables[0]
    assert type(groupby_agg_node) is SeriesData
    assert type(groupby_agg_node.op) is DataFrameGroupByAgg
    assert groupby_agg_node.name == "c2"

    groupby_agg_node_preds = graph.predecessors(groupby_agg_node)
    assert len(groupby_agg_node_preds) == 1
    read_csv_node = groupby_agg_node_preds[0]
    assert type(read_csv_node) is DataFrameData
    assert type(read_csv_node.op) is DataFrameReadCSV
    assert len(read_csv_node.op.usecols) == 2
    assert len({"c1", "c2"} ^ set(read_csv_node.op.usecols)) == 0

    raw = pd.read_csv(file_path)
    pd_res = raw.groupby("c1")["c2"].sum()
    r = c.execute().fetch()
    pd.testing.assert_series_equal(r, pd_res)


def test_merge(setup, gen_data1):
    file_path, file_path2 = gen_data1
    df1 = md.read_csv(file_path)
    df2 = md.read_csv(file_path2)
    c = df1.merge(df2, left_on="c1", right_on="c1")["c1"]

    graph = c.build_graph()
    optimize(graph)
    graph.view()

    # merge_node = graph.result_tileables[0]
    # assert type(merge_node.op) is DataFrameMerge
    #
    # merge_node_preds = graph.predecessors(merge_node)
    # assert len(merge_node_preds) == 2
    #
    # read_csv_node = merge_node_preds[0]
    # read_csv_op = read_csv_node.op
    # assert type(read_csv_op) is DataFrameReadCSV
    # assert len(read_csv_op.usecols) == 1
    #
    # graph2 = TileableGraph([df1.data])
    # next(TileableGraphBuilder(graph2).build())
    # optimize(graph2)
    #
    # origin_read_csv_node = graph2.result_tileables[0]
    # assert len(origin_read_csv_node.op.usecols) == 0

    r = c.execute().fetch()
    raw1 = pd.read_csv(file_path)
    raw2 = pd.read_csv(file_path2)
    expected = raw1.merge(raw2, left_on="c1", right_on="c1")["c1"]
    pd.testing.assert_series_equal(r, expected)
