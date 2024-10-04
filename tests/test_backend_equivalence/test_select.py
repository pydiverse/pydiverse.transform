from __future__ import annotations

from pydiverse.transform._internal.pipe.verbs import (
    select,
)
from tests.util import assert_result_equal


def test_simple_select(df1):
    assert_result_equal(df1, lambda t: t >> select(t.col1))
    assert_result_equal(df1, lambda t: t >> select(t.col2))


def test_reorder(df1):
    assert_result_equal(df1, lambda t: t >> select(t.col2, t.col1))
