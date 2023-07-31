from __future__ import annotations

from pydiverse.transform import λ
from pydiverse.transform.core import functions as f
from pydiverse.transform.core.verbs import (
    arrange,
    filter,
    group_by,
    mutate,
    select,
    summarise,
    ungroup,
)
from pydiverse.transform.errors import FunctionTypeError

from . import assert_result_equal, full_sort


def test_simple_ungrouped(df3):
    assert_result_equal(
        df3,
        lambda t: t >> mutate(min=t.col4.min(), max=t.col4.max(), mean=t.col4.mean()),
    )


def test_simple_grouped(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(min=t.col4.min(), max=t.col4.max(), mean=t.col4.mean()),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(min=t.col4.min(), max=t.col4.max(), mean=t.col4.mean()),
    )


def test_chained(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(min=t.col4.min())
        >> mutate(max=t.col4.max(), mean=t.col4.mean()),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(min=t.col4.min(), max=t.col4.max())
        >> mutate(span=λ.max - λ.min),
    )


def test_nested(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(range=t.col4.max() - 10)
        >> ungroup()
        >> mutate(range_mean=λ.range.mean()),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(x=λ.col4.max())
        >> mutate(y=λ.x.min() * 1)
        >> mutate(z=λ.y.mean())
        >> mutate(w=λ.x / λ.y),
    )

    assert_result_equal(
        df3,
        lambda t: t >> mutate(x=(λ.col4.max().min() + λ.col2.mean()).max()),
        exception=FunctionTypeError,
        may_throw=True,
    )


def test_filter(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(mean3=t.col3.mean())
        >> filter(λ.mean3 <= 2.0),
    )


def test_arrange(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(mean3=t.col3.mean())
        >> arrange(λ.mean3),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(-t.col4)
        >> group_by(t.col1, t.col2)
        >> mutate(mean3=t.col3.mean())
        >> arrange(λ.mean3),
    )


def test_summarise(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(range=t.col4.max() - t.col4.min())
        >> summarise(mean_range=λ.range.mean()),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(range=t.col4.max() - t.col4.min())
        >> mutate(mean_range=λ.range.mean()),
    )


def test_intermediate_select(df3):
    # Check that subqueries happen transparently
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(x=t.col4.mean())
        >> select()
        >> mutate(y=λ.x.min())
        >> select()
        >> mutate(z=(λ.x - λ.y).mean()),
    )


def test_arrange_argument(df3):
    # Grouped
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(x=λ.col4.shift(1, arrange=[-λ.col3]))
        >> full_sort()
        >> select(λ.x),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col2)
        >> mutate(x=f.row_number(arrange=[-λ.col4]))
        >> full_sort()
        >> select(λ.x),
    )

    # Ungrouped
    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(x=λ.col4.shift(1, arrange=[-λ.col3]))
        >> full_sort()
        >> select(λ.x),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(x=f.row_number(arrange=[-λ.col4]))
        >> full_sort()
        >> select(λ.x),
    )


def test_complex(df3):
    # Window function before summarise
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(mean3=t.col3.mean(), rn=f.row_number(arrange=[λ.col1, λ.col2]))
        >> filter(λ.mean3 > λ.rn)
        >> summarise(meta_mean=λ.mean3.mean())
        >> filter(t.col1 >= λ.meta_mean)
        >> filter(t.col1 != 1)
        >> arrange(λ.meta_mean),
    )

    # Window function after summarise
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> mutate(minM3=λ.mean3.min(), maxM3=λ.mean3.max())
        >> mutate(span=λ.maxM3 - λ.minM3)
        >> filter(λ.span < 3)
        >> arrange(λ.span),
    )


def test_nested_bool(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(x=t.col1 <= t.col2, y=(t.col3 * 4) >= λ.col4)
        >> mutate(
            xshift=λ.x.shift(1, arrange=[t.col4]),
            yshift=λ.y.shift(-1, arrange=[t.col4]),
        )
        >> mutate(xAndY=λ.x & λ.y, xAndYshifted=λ.xshift & λ.yshift),
    )


# Test specific operations


def test_op_shift(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(
            shift1=t.col2.shift(1, arrange=[t.col4]),
            shift2=t.col4.shift(-2, 0, arrange=[t.col4]),
            shift3=t.col4.shift(0, arrange=[t.col4]),
        ),
    )


def test_op_row_number(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(
            row_number1=f.row_number(arrange=[λ.col4]),
            row_number2=f.row_number(arrange=[λ.col2, λ.col3]),
        ),
    )


def test_op_rank(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(
            rank1=t.col1.rank(),
            rank2=t.col2.rank(),
            rank3=t.col2.nulls_last().rank(),
            rank4=t.col5.nulls_first().rank(),
            rank5=(-t.col5.nulls_first()).rank(),
            rank_expr=(t.col3 - t.col2).rank(),
        ),
    )


def test_op_dense_rank(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(
            rank1=t.col1.dense_rank(),
            rank2=t.col2.dense_rank(),
            rank3=t.col2.nulls_last().dense_rank(),
            rank4=t.col5.nulls_first().dense_rank(),
            rank5=(-t.col5.nulls_first()).dense_rank(),
        ),
    )
