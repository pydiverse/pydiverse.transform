# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pydiverse.transform.extended import *
from tests.util.assertion import assert_result_equal


def test_dtypes(df1):
    assert_result_equal(
        df1,
        lambda t: t >> filter(t.col1 % 2 == 1) >> inner_join(s := t >> mutate(u=t.col1 % 2) >> alias(), t.col1 == s.u),
    )
