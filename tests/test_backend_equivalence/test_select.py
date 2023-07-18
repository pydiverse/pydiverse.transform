from __future__ import annotations

from pydiverse.transform import λ
from pydiverse.transform.core.verbs import (
    mutate,
    select,
)

from . import assert_result_equal, tables


@tables("df1")
def test_simple_select(df1_x, df1_y):
    assert_result_equal(df1_x, df1_y, lambda t: t >> select(t.col1))
    assert_result_equal(df1_x, df1_y, lambda t: t >> select(t.col2))


@tables("df1")
def test_reorder(df1_x, df1_y):
    assert_result_equal(df1_x, df1_y, lambda t: t >> select(t.col2, t.col1))


@tables("df3")
def test_ellipsis(df3_x, df3_y):
    assert_result_equal(df3_x, df3_y, lambda t: t >> select(...))
    assert_result_equal(df3_x, df3_y, lambda t: t >> select(t.col1) >> select(...))
    assert_result_equal(
        df3_x, df3_y, lambda t: t >> mutate(x=t.col1 * 2) >> select() >> select(...)
    )


@tables("df3")
def test_negative_select(df3_x, df3_y):
    assert_result_equal(df3_x, df3_y, lambda t: t >> select(-t.col1))
    assert_result_equal(df3_x, df3_y, lambda t: t >> select(-λ.col1, -t.col2))
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t >> select() >> mutate(x=t.col1 * 2) >> select(-λ.col3),
    )
