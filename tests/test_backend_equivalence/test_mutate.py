from __future__ import annotations

import pydiverse.transform as pdt
from pydiverse.transform import C
from pydiverse.transform._internal.pipe.verbs import (
    mutate,
    select,
)
from tests.util import assert_result_equal


def test_noop(df2):
    assert_result_equal(
        df2, lambda t: t >> mutate(col1=t.col1, col2=t.col2, col3=t.col3)
    )


def test_multiply(df1):
    assert_result_equal(df1, lambda t: t >> mutate(x2=t.col1 * 2))
    assert_result_equal(df1, lambda t: t >> select() >> mutate(x2=t.col1 * 2))


def test_reorder(df2):
    assert_result_equal(df2, lambda t: t >> mutate(col1=t.col2, col2=t.col1))

    assert_result_equal(
        df2,
        lambda t: t
        >> mutate(col1=t.col2, col2=t.col1)
        >> mutate(col1=t.col2, col2=C.col3, col3=C.col2),
    )


def test_literals(df1):
    assert_result_equal(df1, lambda t: t >> mutate(x=1))
    assert_result_equal(df1, lambda t: t >> mutate(x=1.1))
    assert_result_equal(df1, lambda t: t >> mutate(x=True))
    assert_result_equal(df1, lambda t: t >> mutate(x="test"))
    assert_result_equal(df1, lambda t: t >> mutate(u=pdt.lit(None)))


def test_none(df4):
    assert_result_equal(df4, lambda t: t >> mutate(x=None))
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x1=t.col1.is_null(),
            y1=~t.col2.is_null(),
        ),
    )


def test_mutate_bool_expr(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(x=t.col1 <= t.col2, y=(t.col3 * 4) >= C.col4)
        >> mutate(xAndY=C.x & C.y),
    )
