from __future__ import annotations

from pydiverse.transform import C
from pydiverse.transform.core import functions as f
from pydiverse.transform.core.verbs import (
    arrange,
    filter,
    group_by,
    left_join,
    mutate,
    select,
    slice_head,
    summarise,
)
from tests.util import assert_result_equal, full_sort


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
        lambda t: t >> arrange(*t) >> slice_head(1) >> arrange(*t) >> slice_head(1),
    )
    assert_result_equal(
        df3,
        lambda t: t >> arrange(*t) >> slice_head(10) >> arrange(*t) >> slice_head(5),
    )
    assert_result_equal(
        df3,
        lambda t: t >> arrange(*t) >> slice_head(100) >> arrange(*t) >> slice_head(5),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(*t)
        >> slice_head(2, offset=5)
        >> arrange(*t)
        >> slice_head(2, offset=1),
    )
    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(*t)
        >> slice_head(10, offset=8)
        >> arrange(*t)
        >> slice_head(10, offset=1),
    )


def test_with_select(df3):
    assert_result_equal(
        df3,
        lambda t: t >> select() >> arrange(*t) >> slice_head(4, offset=2) >> select(*t),
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
        >> full_sort()
        >> arrange(*t)
        >> slice_head(3)
        >> left_join(u, t.col1 == u.col1)
        >> full_sort(),
    )

    assert_result_equal(
        (df1, df2),
        lambda t, u: t
        >> left_join(u >> arrange(*t) >> slice_head(2, offset=1), t.col1 == u.col1)
        >> full_sort(),
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
        lambda t: t >> arrange(*t) >> slice_head(4, offset=2) >> filter(t.col1 == 1),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> filter(t.col4 % 2 == 0)
        >> arrange(*t)
        >> slice_head(4, offset=2)
        >> filter(t.col1 == 1),
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
        >> arrange(-C.x, C.col5),
    )


def test_with_group_by(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(*t)
        >> slice_head(1)
        >> group_by(C.col1)
        >> mutate(x=f.count()),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(C.col1, *t)
        >> slice_head(6, offset=1)
        >> group_by(C.col1)
        >> select()
        >> mutate(x=C.col4.mean()),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(key=C.col4 % (C.col3 + 1))
        >> arrange(C.key, *t)
        >> slice_head(4)
        >> group_by(C.key)
        >> summarise(x=f.count()),
    )


def test_with_summarise(df3):
    assert_result_equal(
        df3,
        lambda t: t >> arrange(*t) >> slice_head(4) >> summarise(count=f.count()),
    )

    assert_result_equal(
        df3,
        lambda t: t >> arrange(*t) >> slice_head(4) >> summarise(c3_mean=C.col3.mean()),
    )
