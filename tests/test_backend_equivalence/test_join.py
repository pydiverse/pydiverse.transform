from __future__ import annotations

import sqlite3

import pytest

from pydiverse.transform.core.verbs import (
    alias,
    join,
    select,
)

from . import assert_result_equal, full_sort


@pytest.mark.parametrize(
    "how",
    [
        "inner",
        "left",
        pytest.param(
            "outer",
            marks=pytest.mark.skipif(
                sqlite3.sqlite_version < "3.39.0",
                reason="SQLite version doesn't support OUTER JOIN",
            ),
        ),
    ],
)
def test_join(df1, df2, how):
    assert_result_equal(
        (df1, df2),
        lambda t, u: t >> join(u, t.col1 == u.col1, how=how) >> full_sort(),
    )

    assert_result_equal(
        (df1, df2),
        lambda t, u: t
        >> join(u, (t.col1 == u.col1) & (t.col1 == u.col2), how=how)
        >> full_sort(),
    )


@pytest.mark.parametrize(
    "how",
    [
        "inner",
        "left",
        pytest.param(
            "outer",
            marks=pytest.mark.skipif(
                sqlite3.sqlite_version < "3.39.0",
                reason="SQLite version doesn't support OUTER JOIN",
            ),
        ),
    ],
)
def test_join_and_select(df1, df2, how):
    assert_result_equal(
        (df1, df2),
        lambda t, u: t >> select() >> join(u, t.col1 == u.col1, how=how) >> full_sort(),
    )

    assert_result_equal(
        (df1, df2),
        lambda t, u: t
        >> join(u >> select(), (t.col1 == u.col1) & (t.col1 == u.col2), how=how)
        >> full_sort(),
    )


@pytest.mark.parametrize(
    "how",
    [
        "inner",
        "left",
        pytest.param(
            "outer",
            marks=pytest.mark.skipif(
                sqlite3.sqlite_version < "3.39.0",
                reason="SQLite version doesn't support OUTER JOIN",
            ),
        ),
    ],
)
def test_self_join(df3, how):
    # Self join without alias should raise an exception
    assert_result_equal(
        df3,
        lambda t: t >> join(t, t.col1 == t.col1, how=how),
        exception=ValueError,
    )

    def self_join_1(t):
        u = t >> alias("self_join")
        return t >> join(u, t.col1 == u.col1, how=how) >> full_sort()

    assert_result_equal(df3, self_join_1)

    def self_join_2(t):
        u = t >> alias("self_join")
        return (
            t
            >> join(u, (t.col1 == u.col1) & (t.col2 == u.col2), how=how)
            >> full_sort()
        )

    assert_result_equal(df3, self_join_2)

    def self_join_3(t):
        u = t >> alias("self_join")
        return t >> join(u, (t.col2 == u.col3), how=how) >> full_sort()

    assert_result_equal(df3, self_join_3)
