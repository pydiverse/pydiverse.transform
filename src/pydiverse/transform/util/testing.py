from __future__ import annotations

import pandas as pd

from pydiverse.transform.core import Table
from pydiverse.transform.core.verbs import collect, show_query


def assert_equal(left, right, check_dtype=False):
    left_df = left >> collect() if isinstance(left, Table) else left
    right_df = right >> collect() if isinstance(right, Table) else right

    try:
        pd.testing.assert_frame_equal(left_df, right_df, check_dtype=check_dtype)
    except AssertionError as e:
        print("First dataframe:")
        print(left_df)
        if isinstance(left, Table):
            left >> show_query()
        print()
        print("Second dataframe:")
        print(right_df)
        if isinstance(right, Table):
            right >> show_query()
        raise e
