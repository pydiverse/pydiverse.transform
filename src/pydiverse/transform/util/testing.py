from __future__ import annotations

import pandas as pd

from pydiverse.transform.core import Table
from pydiverse.transform.core.verbs import collect


def assert_equal(left, right, check_dtype=False):
    if isinstance(left, Table):
        left = left >> collect()
    if isinstance(right, Table):
        right = right >> collect()

    pd.util.testing.assert_frame_equal(left, right, check_dtype=check_dtype)
