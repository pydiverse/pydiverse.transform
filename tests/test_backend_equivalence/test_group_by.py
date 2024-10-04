from __future__ import annotations

import pytest

from pydiverse.transform import C
from pydiverse.transform._internal.pipe import functions
from pydiverse.transform._internal.pipe.verbs import (
    arrange,
    filter,
    group_by,
    join,
    left_join,
    mutate,
    select,
    ungroup,
)
from tests.util import assert_result_equal


def test_ungroup(df3):
    assert_result_equal(df3, lambda t: t >> group_by(t.col1, t.col2) >> ungroup())


def test_select(df3):
    assert_result_equal(
        df3,
        lambda t: t >> group_by(t.col1, t.col2) >> select(t.col1, t.col3),
    )


def test_mutate(df3, df4):
    assert_result_equal(
        df3,
        lambda t: t >> mutate(c1xc2=t.col1 * t.col2) >> group_by(C.c1xc2),
    )

    assert_result_equal(
        df3,
        lambda t: t >> group_by(t.col1, t.col2) >> mutate(c1xc2=t.col1 * t.col2),
    )

    assert_result_equal(
        (df3, df4),
        lambda t, u: t
        >> group_by(t.col1, t.col2)
        >> mutate(col1=t.col1 * t.col2)
        >> arrange(t.col3.descending().nulls_last())
        >> ungroup()
        >> left_join(u, t.col2 == u.col2)
        >> mutate(
            x=functions.row_number(
                arrange=[u.col4.nulls_last(), t.col4.nulls_first()],
                partition_by=[t.col1],
            ),
            p=t.col1 * u.col4,
            y=functions.rank(
                arrange=[(t.col1 * u.col4).nulls_last().nulls_first().nulls_last()]
            ),
        ),
    )


def test_grouped_join(df1, df3):
    # Joining a grouped table should always throw an exception
    assert_result_equal(
        (df1, df3),
        lambda t, u: t >> group_by(C.col1) >> join(u, t.col1 == u.col1, how="left"),
        exception=ValueError,
    )

    assert_result_equal(
        (df1, df3),
        lambda t, u: t >> join(u >> group_by(C.col1), t.col1 == u.col1, how="left"),
        exception=ValueError,
    )


@pytest.mark.parametrize("how", ["inner", "left"])
def test_ungrouped_join(df1, df3, how):
    # After ungrouping joining should work again
    assert_result_equal(
        (df1, df3),
        lambda t, u: t
        >> group_by(t.col1)
        >> ungroup()
        >> join(u, t.col1 == u.col1, how=how),
        check_row_order=False,
    )


def test_filter(df3):
    assert_result_equal(df3, lambda t: t >> group_by(t.col1) >> filter(t.col3 >= 2))


def test_arrange(df3):
    assert_result_equal(
        df3, lambda t: t >> group_by(t.col1) >> arrange(t.col1, -t.col3)
    )

    assert_result_equal(df3, lambda t: t >> group_by(t.col1) >> arrange(-t.col4))


def test_group_by_bool_col(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(x=t.col1 <= t.col2)
        >> group_by(C.x)
        >> mutate(y=C.col4.mean()),
    )
