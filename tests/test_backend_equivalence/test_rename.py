from __future__ import annotations

from pydiverse.transform.core.verbs import (
    rename,
)

from . import assert_result_equal, tables


@tables("df3")
def test_noop(df3_x, df3_y):
    assert_result_equal(df3_x, df3_y, lambda t: t >> rename({}))
    assert_result_equal(df3_x, df3_y, lambda t: t >> rename({"col1": "col1"}))


@tables("df3")
def test_simple(df3_x, df3_y):
    assert_result_equal(df3_x, df3_y, lambda t: t >> rename({"col1": "X"}))
    assert_result_equal(df3_x, df3_y, lambda t: t >> rename({"col2": "Y"}))
    assert_result_equal(df3_x, df3_y, lambda t: t >> rename({"col1": "A", "col2": "B"}))
    assert_result_equal(df3_x, df3_y, lambda t: t >> rename({"col2": "B", "col1": "A"}))


@tables("df3")
def test_chained(df3_x, df3_y):
    assert_result_equal(
        df3_x, df3_y, lambda t: t >> rename({"col1": "X"}) >> rename({"X": "Y"})
    )
    assert_result_equal(
        df3_x, df3_y, lambda t: t >> rename({"col1": "X"}) >> rename({"X": "col1"})
    )
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> rename({"col1": "1", "col2": "2"})
        >> rename({"1": "col1", "2": "col2"}),
    )


@tables("df3")
def test_complex(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t >> rename({"col1": "col2", "col2": "col3", "col3": "col1"}),
    )
