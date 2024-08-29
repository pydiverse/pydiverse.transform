from __future__ import annotations

from pydiverse.transform import C
from pydiverse.transform.core.verbs import (
    arrange,
    filter,
    group_by,
    mutate,
    select,
    summarise,
)
from pydiverse.transform.errors import ExpressionTypeError, FunctionTypeError
from tests.util import assert_result_equal


def test_ungrouped(df3):
    assert_result_equal(df3, lambda t: t >> summarise(mean3=t.col3.mean()))
    assert_result_equal(
        df3,
        lambda t: t >> summarise(mean3=t.col3.mean(), mean4=t.col4.mean()),
    )


def test_simple_grouped(df3):
    assert_result_equal(
        df3,
        lambda t: t >> group_by(t.col1) >> summarise(mean3=t.col3.mean()),
    )


def test_multi_grouped(df3):
    assert_result_equal(
        df3,
        lambda t: t >> group_by(t.col1, t.col2) >> summarise(mean3=t.col3.mean()),
    )


def test_chained_summarised(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> summarise(mean_of_mean3=C.mean3.mean()),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(k=(C.col1 + C.col2) * C.col4)
        >> group_by(C.k)
        >> summarise(x=C.col4.mean())
        >> summarise(y=C.k.mean()),
    )


def test_nested(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean_of_mean3=t.col3.mean().mean()),
        exception=FunctionTypeError,
    )


def test_select(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> select(t.col1, C.mean3, t.col2),
    )


def test_mutate(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> mutate(x10=C.mean3 * 10),
    )


def test_filter(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> filter(C.mean3 <= 2.0),
    )


# def test_filter_argument(df3):
#     assert_result_equal(
#         df3,
#         lambda t: t
#         >> group_by(t.col2)
#         >> summarise(u=t.col4.sum(filter=(t.col1 != 0))),
#     )

#     assert_result_equal(
#         df3,
#         lambda t: t
#         >> group_by(t.col4, t.col1)
#         >> summarise(
#             u=(t.col3 * t.col4 - t.col2).sum(
#                 filter=(t.col5.isin("a", "e", "i", "o", "u"))
#             )
#         ),
#     )


def test_arrange(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> arrange(C.mean3),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(-t.col4)
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> arrange(C.mean3),
    )


def test_intermediate_select(df3):
    # Check that subqueries happen transparently
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(x=t.col4.mean())
        >> mutate(x2=C.x * 2)
        >> select()
        >> summarise(y=(C.x - C.x2).min()),
    )


def test_not_summarising(df4):
    assert_result_equal(
        df4, lambda t: t >> summarise(x=C.col1), exception=FunctionTypeError
    )


def test_none(df4):
    assert_result_equal(
        df4, lambda t: t >> summarise(x=None), exception=ExpressionTypeError
    )


# TODO: Implement more test cases for summarise verb


# Test specific operations


def test_op_min(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(t.col1)
        >> summarise(**{c._.name + "_min": c.min() for c in t}),
    )


def test_op_max(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(t.col1)
        >> summarise(**{c._.name + "_max": c.max() for c in t}),
    )


def test_op_any(df4):
    assert_result_equal(
        df4,
        lambda t: t >> group_by(t.col1) >> summarise(any=(C.col1 == C.col2).any()),
    )
    assert_result_equal(
        df4,
        lambda t: t >> group_by(t.col1) >> mutate(any=(C.col1 == C.col2).any()),
    )


def test_op_all(df4):
    assert_result_equal(
        df4,
        lambda t: t >> group_by(t.col1) >> summarise(all=(C.col2 != C.col3).all()),
    )
    assert_result_equal(
        df4,
        lambda t: t >> group_by(t.col1) >> mutate(all=(C.col2 != C.col3).all()),
    )
