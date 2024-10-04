from __future__ import annotations

import pydiverse.transform as pdt
from pydiverse.transform import C
from pydiverse.transform._internal.pipe.verbs import (
    alias,
    arrange,
    filter,
    group_by,
    left_join,
    mutate,
    select,
    slice_head,
    summarize,
)
from tests.util import assert_result_equal


def test_simple(df3):
    assert_result_equal(df3, lambda t: t >> arrange(*t) >> slice_head(1))
    assert_result_equal(df3, lambda t: t >> arrange(*t) >> slice_head(10))
    assert_result_equal(df3, lambda t: t >> arrange(*t) >> slice_head(100))

    assert_result_equal(df3, lambda t: t >> arrange(*t) >> slice_head(1, offset=8))
    assert_result_equal(df3, lambda t: t >> arrange(*t) >> slice_head(10, offset=8))
    assert_result_equal(df3, lambda t: t >> arrange(*t) >> slice_head(100, offset=8))

    assert_result_equal(df3, lambda t: t >> arrange(*t) >> slice_head(1, offset=100))
    assert_result_equal(df3, lambda t: t >> arrange(*t) >> slice_head(10, offset=100))
    assert_result_equal(df3, lambda t: t >> arrange(*t) >> slice_head(100, offset=100))


def test_chained(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(*t)
        >> slice_head(1)
        >> alias()
        >> (lambda s: arrange(*s) >> slice_head(1)),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(*t)
        >> slice_head(10)
        >> alias()
        >> (lambda s: arrange(*s) >> slice_head(5)),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(*t)
        >> slice_head(2, offset=5)
        >> alias()
        >> (lambda s: arrange(*s) >> slice_head(2, offset=1)),
    )


def test_with_mutate(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(a=C.col1 * 2)
        >> arrange(*t)
        >> slice_head(4, offset=2)
        >> mutate(b=C.col2 + C.a),
    )


def test_with_join(df1, df2):
    assert_result_equal(
        (df1, df2),
        lambda t, u: t
        >> arrange(*t)
        >> slice_head(3)
        >> alias()
        >> left_join(u, C.col1 == u.col1),
        check_row_order=False,
    )

    assert_result_equal(
        (df1, df2),
        lambda t, u: t
        >> left_join(u >> arrange(*t) >> slice_head(2, offset=1), t.col1 == u.col1),
        check_row_order=False,
        exception=ValueError,
        may_throw=True,
    )


def test_with_filter(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> filter(t.col4 % 2 == 0)
        >> arrange(*t)
        >> slice_head(4, offset=2),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(*t)
        >> slice_head(4, offset=2)
        >> alias()
        >> filter(C.col1 == 1),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> filter(t.col4 % 2 == 0)
        >> arrange(*t)
        >> slice_head(4, offset=2)
        >> alias()
        >> filter(C.col1 == 1),
    )


def test_with_arrange(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(x=t.col4 - (t.col1 * t.col2))
        >> arrange(C.x, *t)
        >> slice_head(4, offset=2),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(x=(t.col1 * t.col2))
        >> arrange(*t)
        >> slice_head(4)
        >> alias()
        >> (lambda _: arrange(-C.x, C.col5)),
    )


def test_with_group_by(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(*t)
        >> slice_head(1)
        >> alias()
        >> group_by(C.col1)
        >> mutate(x=pdt.count()),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(C.col1, *t)
        >> slice_head(6, offset=1)
        >> alias()
        >> group_by(C.col1)
        >> mutate(x=C.col4.mean())
        >> select(C.x),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(key=C.col4 % (C.col3 + 1))
        >> arrange(C.key, *t)
        >> slice_head(4)
        >> alias()
        >> group_by(C.key)
        >> summarize(x=pdt.count()),
    )


def test_with_summarize(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(*t)
        >> slice_head(4)
        >> alias()
        >> summarize(count=pdt.count()),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(*t)
        >> slice_head(4)
        >> alias()
        >> summarize(c3_mean=C.col3.mean()),
    )
