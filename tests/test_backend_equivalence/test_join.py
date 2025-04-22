from __future__ import annotations

import sqlite3

import pytest

import pydiverse.transform as pdt
from pydiverse.transform.errors import FunctionTypeError
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

    assert_result_equal(
        (df3, df4), lambda s, t: s >> inner_join(t, on=["col1", s.col2 <= t.col2])
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

    assert_result_equal(
        (df3, df4),
        lambda t3, t4: t3
        >> summarize(y=t3.col1.max(), z=t3.col4.mean())
        >> alias()
        >> left_join(t4, on=C.y == t4.col4),
    )


def test_join_window(df3, df4):
    assert_result_equal(
        (df3, df4),
        lambda t3, t4: t3
        >> mutate(y=t3.col1.dense_rank())
        >> alias()
        >> inner_join(t4, on=C.y == t4.col1),
    )

    assert_result_equal(
        (df3, df4),
        lambda s, t: s
        >> mutate(y=s.col1.shift(1, arrange=s.col4.nulls_first()))
        >> alias()
        >> inner_join(t, on="col2"),
    )

    assert_result_equal(
        (df3, df4),
        lambda t3, t4: t3 >> inner_join(t4, on=t3.col1.dense_rank() == t4.col1),
        exception=FunctionTypeError,
    )


def test_join_where(df2, df3, df4):
    assert_result_equal(
        (df2, df3),
        lambda t2, t3: t2 >> left_join(t3 >> filter(t3.col4 >= 2), on="col1"),
    )

    assert_result_equal(
        (df3, df4),
        lambda t3, t4: t3
        >> filter(t3.col4 != -1729)
        >> left_join(t4 >> filter(t4.col3 > 0), on=t3.col2 == t4.col2),
    )

    assert_result_equal(
        (df3, df4),
        lambda t3, t4: t3
        >> filter(t3.col1 % 2 == 0)
        >> alias()
        >> full_join(
            t4_filtered := t4 >> filter(t4.col1.is_not_null()) >> alias(),
            on=C.col2 == t4_filtered.col2,
        ),
    )


def test_join_const_col(df3, df4):
    assert_result_equal(
        (df3, df4),
        lambda s, t: s >> left_join(t >> mutate(y=0) >> alias(), on=s.col1 == C.y_df4),
    )

    assert_result_equal(
        (df3, df4),
        lambda s, t: s
        >> mutate(z=2)
        >> alias()
        >> full_join(t >> mutate(j=True) >> alias(), on="col2"),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(x=4, y=5)
        >> mutate(z=C.x + C.y)
        >> alias()
        >> full_join(t_ := t >> alias(), on=C.col1 == t_.col2),
    )


def test_cross_join(df2, df3):
    assert_result_equal((df2, df3), lambda s, t: s >> cross_join(t))
