# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pydiverse.transform as pdt
from pydiverse.transform.common import *
from tests.util import assert_result_equal


def test_lambda_cols(df3):
    assert_result_equal(df3, lambda t: t >> select(C.col1, C.col2))
    assert_result_equal(df3, lambda t: t >> mutate(col1=C.col1, col2=C.col1))
    assert_result_equal(df3, lambda t: t >> select(C.col10), exception=pdt.ColumnNotFoundError)


def test_transfer_col_references(df3, df4):
    @pdt.verb
    def collect_with_refs(tbl):
        if pdt.is_sql_backed(tbl):
            return tbl
        return pdt.transfer_col_references(tbl >> collect(keep_col_refs=False), tbl)

    assert_result_equal(df3, lambda t: pdt.transfer_col_references(t, t))
    assert_result_equal(df3, lambda t: t >> collect_with_refs() >> mutate(z=t.col1 + t.col2))
    assert_result_equal(
        (df3, df4),
        lambda t, u: pdt.transfer_col_references(u, t) >> mutate(s=t.col1 * t.col4),
    )
