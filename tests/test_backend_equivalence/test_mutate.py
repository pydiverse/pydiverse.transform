from __future__ import annotations

from pydiverse.transform import λ
from pydiverse.transform.core.verbs import (
    mutate,
    select,
)
from pydiverse.transform.errors import ExpressionTypeError
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
        >> mutate(col1=t.col2, col2=λ.col3, col3=λ.col2),
    )


def test_literals(df1):
    assert_result_equal(df1, lambda t: t >> mutate(x=1))
    assert_result_equal(df1, lambda t: t >> mutate(x=1.1))
    assert_result_equal(df1, lambda t: t >> mutate(x=True))
    assert_result_equal(df1, lambda t: t >> mutate(x="test"))


def test_none(df4):
    assert_result_equal(
        df4, lambda t: t >> mutate(x=None), exception=ExpressionTypeError
    )
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            x1=(t.col1 == None),
            x2=(t.col1 != None),
            y1=(None == t.col2),
            y2=(None != t.col2),
        ),
    )


def test_mutate_bool_expr(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(x=t.col1 <= t.col2, y=(t.col3 * 4) >= λ.col4)
        >> mutate(xAndY=λ.x & λ.y),
    )
