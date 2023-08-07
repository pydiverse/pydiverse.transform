from __future__ import annotations

import pytest

from pydiverse.transform import λ
from pydiverse.transform.core.verbs import (
    arrange,
    filter,
    group_by,
    join,
    mutate,
    select,
    ungroup,
)
from tests.util import assert_result_equal, full_sort


def test_ungroup(df3):
    assert_result_equal(df3, lambda t: t >> group_by(t.col1, t.col2) >> ungroup())


def test_select(df3):
    assert_result_equal(
        df3,
        lambda t: t >> group_by(t.col1, t.col2) >> select(t.col1, t.col3),
    )


def test_mutate(df3):
    assert_result_equal(
        df3,
        lambda t: t >> mutate(c1xc2=t.col1 * t.col2) >> group_by(λ.c1xc2),
    )

    assert_result_equal(
        df3,
        lambda t: t >> group_by(t.col1, t.col2) >> mutate(c1xc2=t.col1 * t.col2),
    )

    assert_result_equal(
        df3,
        lambda t: t >> group_by(t.col1, t.col2) >> mutate(col1=t.col1 * t.col2),
    )


def test_grouped_join(df1, df3):
    # Joining a grouped table should always throw an exception
    assert_result_equal(
        (df1, df3),
        lambda t, u: t >> group_by(λ.col1) >> join(u, t.col1 == u.col1, how="left"),
        exception=ValueError,
    )

    assert_result_equal(
        (df1, df3),
        lambda t, u: t >> join(u >> group_by(λ.col1), t.col1 == u.col1, how="left"),
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
        >> join(u, t.col1 == u.col1, how=how)
        >> full_sort(),
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
        >> group_by(λ.x)
        >> mutate(y=λ.col4.mean()),
    )
