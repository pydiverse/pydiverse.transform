from __future__ import annotations

from pydiverse.transform import C
from pydiverse.transform._internal.pipe.verbs import (
    arrange,
    mutate,
)
from tests.util import assert_result_equal


def test_noop(df1):
    assert_result_equal(
        df1, lambda t: t >> arrange(), may_throw=True, exception=TypeError
    )


def test_arrange(df2):
    assert_result_equal(df2, lambda t: t >> arrange(t.col1.ascending()))
    assert_result_equal(df2, lambda t: t >> arrange((-t.col1).descending()))
    assert_result_equal(df2, lambda t: t >> arrange(t.col3))
    assert_result_equal(df2, lambda t: t >> arrange(-t.col3))


def test_arrange_expression(df3):
    assert_result_equal(
        df3, lambda t: t >> arrange(t.col2, t.col4), check_row_order=True
    )
    assert_result_equal(df3, lambda t: t >> arrange(-t.col4 * 2), check_row_order=True)


def test_arrange_null(df2):
    assert_result_equal(df2, lambda t: t >> arrange(t.col2))
    assert_result_equal(df2, lambda t: t >> arrange(-t.col2))


def test_multiple(df3):
    assert_result_equal(df3, lambda t: t >> arrange(t.col2, -t.col3, -t.col4))

    assert_result_equal(
        df3, lambda t: t >> arrange(t.col2) >> arrange(-t.col3, -t.col4)
    )

    assert_result_equal(
        df3,
        lambda t: t >> arrange(t.col2, -t.col2),
    )

    assert_result_equal(
        df3,
        lambda t: t >> arrange(t.col2, C.col2),
    )


def test_nulls_first(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> arrange(
            t.col1.nulls_first(),
            t.col2.descending().nulls_first(),
            t.col5.nulls_first(),
        ),
        check_row_order=True,
    )


def test_nulls_last(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> arrange(
            t.col1.nulls_last(),
            t.col2.nulls_last().descending(),
            t.col5.nulls_last(),
        ),
        check_row_order=True,
    )


def test_nulls_first_last_mixed(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> arrange(
            t.col1.nulls_first(),
            t.col2.nulls_last().descending(),
            t.col5.descending().nulls_last(),
        ),
        check_row_order=True,
    )


def test_arrange_after_mutate(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(x=t.col1 <= t.col2)
        >> arrange(C.x.nulls_last(), C.col4.nulls_first()),
        check_row_order=True,
    )
