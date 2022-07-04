import pandas as pd

from pdtransform.core import Table
from pdtransform.core.verbs import collect


def assert_equal(left, right, check_dtype=False):
    if isinstance(left, Table):
        left = left >> collect()
    if isinstance(right, Table):
        right = right >> collect()

    pd.util.testing.assert_frame_equal(
        left, right,
        check_dtype=check_dtype
    )
