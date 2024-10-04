from __future__ import annotations

from pydiverse.transform._internal.pipe.verbs import (
    rename,
)
from tests.util import assert_result_equal


def test_noop(df3):
    assert_result_equal(
        df3, lambda t: t >> rename({}), may_throw=True, exception=TypeError
    )
    assert_result_equal(df3, lambda t: t >> rename({"col1": "col1"}))


def test_simple(df3):
    assert_result_equal(df3, lambda t: t >> rename({"col1": "X"}))
    assert_result_equal(df3, lambda t: t >> rename({"col2": "Y"}))
    assert_result_equal(df3, lambda t: t >> rename({"col1": "A", "col2": "B"}))
    assert_result_equal(df3, lambda t: t >> rename({"col2": "B", "col1": "A"}))


def test_chained(df3):
    assert_result_equal(df3, lambda t: t >> rename({"col1": "X"}) >> rename({"X": "Y"}))
    assert_result_equal(
        df3, lambda t: t >> rename({"col1": "X"}) >> rename({"X": "col1"})
    )
    assert_result_equal(
        df3,
        lambda t: t
        >> rename({"col1": "1", "col2": "2"})
        >> rename({"1": "col1", "2": "col2"}),
    )


def test_complex(df3):
    assert_result_equal(
        df3,
        lambda t: t >> rename({"col1": "col2", "col2": "col3", "col3": "col1"}),
    )
