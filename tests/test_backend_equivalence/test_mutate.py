from __future__ import annotations

from pydiverse.transform import λ
from pydiverse.transform.core.verbs import (
    mutate,
    select,
)

from . import assert_result_equal, tables


@tables("df2")
def test_noop(df2_x, df2_y):
    assert_result_equal(
        df2_x, df2_y, lambda t: t >> mutate(col1=t.col1, col2=t.col2, col3=t.col3)
    )


@tables("df1")
def test_multiply(df1_x, df1_y):
    assert_result_equal(df1_x, df1_y, lambda t: t >> mutate(x2=t.col1 * 2))
    assert_result_equal(df1_x, df1_y, lambda t: t >> select() >> mutate(x2=t.col1 * 2))


@tables("df2")
def test_reorder(df2_x, df2_y):
    assert_result_equal(df2_x, df2_y, lambda t: t >> mutate(col1=t.col2, col2=t.col1))

    assert_result_equal(
        df2_x,
        df2_y,
        lambda t: t
        >> mutate(col1=t.col2, col2=t.col1)
        >> mutate(col1=t.col2, col2=λ.col3, col3=λ.col2),
    )


@tables("df1")
def test_literals(df1_x, df1_y):
    assert_result_equal(df1_x, df1_y, lambda t: t >> mutate(x=1))
    assert_result_equal(df1_x, df1_y, lambda t: t >> mutate(x=1.1))
    assert_result_equal(df1_x, df1_y, lambda t: t >> mutate(x=True))
    assert_result_equal(df1_x, df1_y, lambda t: t >> mutate(x="test"))


@tables("df4")
def test_mutate_bool_expr(df4_x, df4_y):
    assert_result_equal(
        df4_x,
        df4_y,
        lambda t: t
        >> mutate(x=t.col1 <= t.col2, y=(t.col3 * 4) >= λ.col4)
        >> mutate(xAndY=λ.x & λ.y),
    )
