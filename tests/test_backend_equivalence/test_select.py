from __future__ import annotations

from pydiverse.transform import C
from pydiverse.transform.core.verbs import (
    mutate,
    select,
)
from tests.util import assert_result_equal


def test_simple_select(df1):
    assert_result_equal(df1, lambda t: t >> select(t.col1))
    assert_result_equal(df1, lambda t: t >> select(t.col2))


def test_reorder(df1):
    assert_result_equal(df1, lambda t: t >> select(t.col2, t.col1))


def test_ellipsis(df3):
    assert_result_equal(df3, lambda t: t >> select(...))
    assert_result_equal(df3, lambda t: t >> select(t.col1) >> select(...))
    assert_result_equal(
        df3, lambda t: t >> mutate(x=t.col1 * 2) >> select() >> select(...)
    )


def test_negative_select(df3):
    assert_result_equal(df3, lambda t: t >> select(-t.col1))
    assert_result_equal(df3, lambda t: t >> select(-C.col1, -t.col2))
    assert_result_equal(
        df3,
        lambda t: t >> select() >> mutate(x=t.col1 * 2) >> select(-C.col3),
    )
