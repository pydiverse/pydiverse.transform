from __future__ import annotations

import sqlite3

import pytest

import pydiverse.transform as pdt
from pydiverse.transform.extended import *
from tests.util import assert_result_equal


@pytest.mark.parametrize(
    "how",
    [
        "inner",
        "left",
        pytest.param(
            "full",
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
        lambda t, u: t >> join(u, t.col1 == u.col1, how=how),
    )

    assert_result_equal(
        (df1, df2),
        lambda t, u: t >> join(u, on="col1", how=how),
    )

    assert_result_equal(
        (df1, df2),
        lambda t, u: t
        >> join(u, on=["col1", t.col1.cast(pdt.Float64()) >= u.col3], how=how),
    )

    assert_result_equal(
        (df1, df2),
        lambda t, u: t
        >> join(u, (t.col1 == u.col1) & (t.col1 == u.col2), how=how)
        >> mutate(l=t.col2.str.len())
        >> left_join(v := u >> alias("v"), v.col2 == C.l)
        >> mutate(k=v.col1 + C.l + u.col2)
        >> full_join(w := t >> alias("w"), (t.col1 == w.col1) & (C.k == w.col1)),
        check_row_order=False,
    )


@pytest.mark.parametrize(
    "how",
    [
        "inner",
        "left",
        pytest.param(
            "full",
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
        lambda t, u: t >> select() >> join(u, t.col1 == u.col1, how=how),
        check_row_order=False,
    )

    assert_result_equal(
        (df1, df2),
        lambda t, u: t
        >> join(u >> select(), (t.col1 == u.col1) & (u.col2 == t.col1), how=how),
        check_row_order=False,
    )


@pytest.mark.parametrize(
    "how",
    [
        "inner",
        "left",
        pytest.param(
            "full",
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
        return t >> join(u, t.col1 == u.col1, how=how)

    assert_result_equal(df3, self_join_1, check_row_order=False)

    def self_join_2(t):
        u = t >> alias("self_join")
        return t >> join(u, (t.col1 == u.col1) & (u.col2 == t.col2), how=how)

    assert_result_equal(df3, self_join_2, check_row_order=False)

    def self_join_3(t):
        u = t >> alias("self_join")
        return t >> join(u, (t.col2 == u.col3), how=how)

    assert_result_equal(df3, self_join_3, check_row_order=False)


def test_ineq_join(df3, df4, df_strings):
    assert_result_equal((df3, df4), lambda t, s: t >> inner_join(s, t.col1 <= s.col1))
    assert_result_equal(
        (df4, df_strings),
        lambda s, t: s
        >> inner_join(
            t >> mutate(u=t.col1.str.len()),
            s.col3 <= C.u_42,
            suffix="_42",
        ),
    )

    assert_result_equal(
        (df3, df_strings),
        lambda s, t: s
        >> inner_join(
            t,
            (s.col1 - s.col2 <= t.c.str.len())
            & (s.col4 >= pdt.when(t.col1.str.starts_with("-")).then(100).otherwise(4)),
        ),
    )

    assert_result_equal(
        (df3, df_strings),
        lambda s, t: s
        >> left_join(
            t,
            (s.col1 - s.col2 <= t.c.str.len() * t.d.str.len())
            & (s.col4 >= pdt.when(t.col1.str.starts_with("-")).then(100).otherwise(7)),
        ),
    )

    assert_result_equal(
        (df3, df_strings),
        lambda s, t: s
        >> left_join(
            t,
            (s.col4 - s.col2 <= t.col1.str.len() * t.d.str.len())
            & (s.col4 >= pdt.when(t.col1.str.starts_with(" ")).then(10).otherwise(7)),
        ),
    )


def test_join_summarize(df3, df4):
    assert_result_equal(
        (df3, df4),
        lambda t3, t4: t3
        >> group_by(t3.col2)
        >> summarize(j=t3.col4.sum())
        >> alias()
        >> inner_join(t4, on="col2"),
    )

    assert_result_equal(
        (df3, df4),
        lambda t3, t4: t4
        >> left_join(
            t3 >> group_by(t3.col2) >> summarize(j=t3.col4.sum()) >> alias(), on="col2"
        ),
    )
