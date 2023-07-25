from __future__ import annotations

from pydiverse.transform import λ
from pydiverse.transform.core.verbs import (
    arrange,
    filter,
    group_by,
    mutate,
    select,
    summarise,
)

from . import assert_result_equal


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
        >> summarise(mean_of_mean3=λ.mean3.mean()),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(k=(λ.col1 + λ.col2) * λ.col4)
        >> group_by(λ.k)
        >> summarise(x=λ.col4.mean())
        >> summarise(y=λ.k.mean()),
    )


def test_nested(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean_of_mean3=t.col3.mean().mean()),
        exception=ValueError,
    )


def test_select(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> select(t.col1, λ.mean3, t.col2),
    )


def test_mutate(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> mutate(x10=λ.mean3 * 10),
    )


def test_filter(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> filter(λ.mean3 <= 2.0),
    )


def test_arrange(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> arrange(λ.mean3),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(-t.col4)
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> arrange(λ.mean3),
    )


def test_intermediate_select(df3):
    # Check that subqueries happen transparently
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(x=t.col4.mean())
        >> mutate(x2=λ.x * 2)
        >> select()
        >> summarise(y=(λ.x - λ.x2).min()),
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
        lambda t: t >> group_by(t.col1) >> summarise(any=(λ.col1 == λ.col2).any()),
    )
    assert_result_equal(
        df4,
        lambda t: t >> group_by(t.col1) >> mutate(any=(λ.col1 == λ.col2).any()),
    )


def test_op_all(df4):
    assert_result_equal(
        df4,
        lambda t: t >> group_by(t.col1) >> summarise(all=(λ.col2 != λ.col3).all()),
    )
    assert_result_equal(
        df4,
        lambda t: t >> group_by(t.col1) >> mutate(all=(λ.col2 != λ.col3).all()),
    )
