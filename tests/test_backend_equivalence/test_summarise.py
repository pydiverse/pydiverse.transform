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

from . import assert_result_equal, tables


@tables("df3")
def test_ungrouped(df3_x, df3_y):
    assert_result_equal(df3_x, df3_y, lambda t: t >> summarise(mean3=t.col3.mean()))
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t >> summarise(mean3=t.col3.mean(), mean4=t.col4.mean()),
    )


@tables("df3")
def test_simple_grouped(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t >> group_by(t.col1) >> summarise(mean3=t.col3.mean()),
    )


@tables("df3")
def test_multi_grouped(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t >> group_by(t.col1, t.col2) >> summarise(mean3=t.col3.mean()),
    )


@tables("df3")
def test_chained_summarised(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> summarise(mean_of_mean3=λ.mean3.mean()),
    )

    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> mutate(k=(λ.col1 + λ.col2) * λ.col4)
        >> group_by(λ.k)
        >> summarise(x=λ.col4.mean())
        >> summarise(y=λ.k.mean()),
    )


@tables("df3")
def test_nested(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean_of_mean3=t.col3.mean().mean()),
        exception=ValueError,
    )


@tables("df3")
def test_select(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> select(t.col1, λ.mean3, t.col2),
    )


@tables("df3")
def test_mutate(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> mutate(x10=λ.mean3 * 10),
    )


@tables("df3")
def test_filter(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> filter(λ.mean3 <= 2.0),
    )


@tables("df3")
def test_arrange(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> arrange(λ.mean3),
    )

    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> arrange(-t.col4)
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> arrange(λ.mean3),
    )


@tables("df3")
def test_intermediate_select(df3_x, df3_y):
    # Check that subqueries happen transparently
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(x=t.col4.mean())
        >> mutate(x2=λ.x * 2)
        >> select()
        >> summarise(y=(λ.x - λ.x2).min()),
    )


# TODO: Implement more test cases for summarise verb


# Test specific operations


@tables("df4")
def test_op_min(df4_x, df4_y):
    assert_result_equal(
        df4_x,
        df4_y,
        lambda t: t
        >> group_by(t.col1)
        >> summarise(**{c._.name + "_min": c.min() for c in t}),
    )


@tables("df4")
def test_op_max(df4_x, df4_y):
    assert_result_equal(
        df4_x,
        df4_y,
        lambda t: t
        >> group_by(t.col1)
        >> summarise(**{c._.name + "_max": c.max() for c in t}),
    )


@tables("df4")
def test_op_any(df4_x, df4_y):
    assert_result_equal(
        df4_x,
        df4_y,
        lambda t: t >> group_by(t.col1) >> summarise(any=(λ.col1 == λ.col2).any()),
    )


@tables("df4")
def test_op_all(df4_x, df4_y):
    assert_result_equal(
        df4_x,
        df4_y,
        lambda t: t >> group_by(t.col1) >> summarise(all=(λ.col2 != λ.col3).all()),
    )
