from __future__ import annotations

from pydiverse.transform import λ
from pydiverse.transform.core.verbs import (
    arrange,
)
from tests.fixtures.backend import skip_backends

from . import assert_result_equal, tables


@tables("df1")
def test_noop(df1_x, df1_y):
    assert_result_equal(df1_x, df1_y, lambda t: t >> arrange())


@tables("df2")
def test_arrange(df2_x, df2_y):
    assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(t.col1))
    assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(-t.col1))
    assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(t.col3))
    assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(-t.col3))


@tables("df2")
def test_arrange_null(df2_x, df2_y):
    assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(t.col2))
    assert_result_equal(df2_x, df2_y, lambda t: t >> arrange(-t.col2))


@tables("df3")
def test_multiple(df3_x, df3_y):
    assert_result_equal(df3_x, df3_y, lambda t: t >> arrange(t.col2, -t.col3, -t.col4))

    assert_result_equal(
        df3_x, df3_y, lambda t: t >> arrange(t.col2) >> arrange(-t.col3, -t.col4)
    )

    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t >> arrange(t.col2, -t.col2),
    )

    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t >> arrange(t.col2, λ.col2),
    )


@tables("df4")
def test_nulls_first(df4_x, df4_y):
    assert_result_equal(
        df4_x,
        df4_y,
        lambda t: t
        >> arrange(
            t.col1.nulls_first(),
            -t.col2.nulls_first(),
            t.col5.nulls_first(),
        ),
        check_order=True,
    )


@tables("df4")
def test_nulls_last(df4_x, df4_y):
    assert_result_equal(
        df4_x,
        df4_y,
        lambda t: t
        >> arrange(
            t.col1.nulls_last(),
            -t.col2.nulls_last(),
            t.col5.nulls_last(),
        ),
        check_order=True,
    )


@skip_backends("pandas")
@tables("df4")
def test_nulls_first_last_mixed(df4_x, df4_y):
    assert_result_equal(
        df4_x,
        df4_y,
        lambda t: t
        >> arrange(
            t.col1.nulls_first(),
            -t.col2.nulls_last(),
            -t.col5,
        ),
        check_order=True,
    )
