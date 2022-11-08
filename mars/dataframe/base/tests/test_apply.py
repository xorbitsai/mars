import pandas as pd
import pytest

from .... import dataframe as md


def test_apply(setup):
    df = pd.DataFrame({"col": [1, 2, 3, 4]})
    mdf = md.DataFrame(df)

    # res = mdf.apply(
    #     lambda x: 20 if x[0] else 10, output_type="df_or_series", axis=1
    # ).execute()
    # print(res.fetch())

    res = mdf.apply(lambda x: x + 1, output_type="df_or_series", axis=1).execute()
    print(res.fetch())

    # res = mdf.apply(lambda x: sum(x), output_type="df_or_series", axis=0).execute()
    # print(res.fetch())
