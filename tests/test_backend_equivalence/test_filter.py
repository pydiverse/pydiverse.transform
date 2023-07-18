from __future__ import annotations

from pydiverse.transform.core.verbs import (
    filter,
)

from . import assert_result_equal, tables


@tables("df2")
def test_noop(df2_x, df2_y):
    assert_result_equal(df2_x, df2_y, lambda t: t >> filter())
    assert_result_equal(df2_x, df2_y, lambda t: t >> filter(t.col1 == t.col1))


@tables("df2")
def test_simple_filter(df2_x, df2_y):
    assert_result_equal(df2_x, df2_y, lambda t: t >> filter(t.col1 == 2))
    assert_result_equal(df2_x, df2_y, lambda t: t >> filter(t.col1 != 2))


@tables("df2")
def test_chained_filters(df2_x, df2_y):
    assert_result_equal(
        df2_x, df2_y, lambda t: t >> filter(1 < t.col1) >> filter(t.col1 < 5)
    )

    assert_result_equal(
        df2_x, df2_y, lambda t: t >> filter(1 < t.col1) >> filter(t.col3 < 0.25)
    )


@tables("df3")
def test_filter_empty_result(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t >> filter(t.col1 == 0) >> filter(t.col2 == 2) >> filter(t.col4 < 2),
    )
