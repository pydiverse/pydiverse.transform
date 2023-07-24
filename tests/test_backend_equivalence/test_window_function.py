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

from . import assert_result_equal, full_sort, tables


@tables("df3")
def test_simple_ungrouped(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t >> mutate(min=t.col4.min(), max=t.col4.max(), mean=t.col4.mean()),
    )


@tables("df3")
def test_simple_grouped(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(min=t.col4.min(), max=t.col4.max(), mean=t.col4.mean()),
    )

    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(min=t.col4.min(), max=t.col4.max(), mean=t.col4.mean()),
    )


@tables("df3")
def test_chained(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(min=t.col4.min())
        >> mutate(max=t.col4.max(), mean=t.col4.mean()),
    )

    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(min=t.col4.min(), max=t.col4.max())
        >> mutate(span=λ.max - λ.min),
    )


@tables("df3")
def test_nested(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(range=t.col4.max() - 10)
        >> ungroup()
        >> mutate(range_mean=λ.range.mean()),
    )

    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> mutate(x=λ.col4.max())
        >> mutate(y=λ.x.min() * 1)
        >> mutate(z=λ.y.mean())
        >> mutate(w=λ.x / λ.y),
    )

    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t >> mutate(x=(λ.col4.max().min() + λ.col2.mean()).max()),
        exception=ValueError,
        may_throw=True,
    )


@tables("df3")
def test_filter(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(mean3=t.col3.mean())
        >> filter(λ.mean3 <= 2.0),
    )


@tables("df3")
def test_arrange(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(mean3=t.col3.mean())
        >> arrange(λ.mean3),
    )

    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> arrange(-t.col4)
        >> group_by(t.col1, t.col2)
        >> mutate(mean3=t.col3.mean())
        >> arrange(λ.mean3),
    )


@tables("df3")
def test_summarise(df3_x, df3_y):
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(range=t.col4.max() - t.col4.min())
        >> summarise(mean_range=λ.range.mean()),
    )

    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(range=t.col4.max() - t.col4.min())
        >> mutate(mean_range=λ.range.mean()),
    )


@tables("df3")
def test_intermediate_select(df3_x, df3_y):
    # Check that subqueries happen transparently
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(x=t.col4.mean())
        >> select()
        >> mutate(y=λ.x.min())
        >> select()
        >> mutate(z=(λ.x - λ.y).mean()),
    )


@tables("df3")
def test_arrange_argument(df3_x, df3_y):
    # Grouped
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(x=λ.col4.shift(1, arrange=[-λ.col3]))
        >> full_sort()
        >> select(λ.x),
    )

    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col2)
        >> mutate(x=f.row_number(arrange=[-λ.col4]))
        >> full_sort()
        >> select(λ.x),
    )

    # Ungrouped
    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> mutate(x=λ.col4.shift(1, arrange=[-λ.col3]))
        >> full_sort()
        >> select(λ.x),
    )

    assert_result_equal(
        df3_x,
        df3_y,
        lambda t: t
        >> mutate(x=f.row_number(arrange=[-λ.col4]))
        >> full_sort()
        >> select(λ.x),
    )


@tables("df3")
def test_complex(df3_x, df3_y):
    # Window function before summarise
    assert_result_equal(
        df3_x,
        df3_y,
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
        df3_x,
        df3_y,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> mutate(minM3=λ.mean3.min(), maxM3=λ.mean3.max())
        >> mutate(span=λ.maxM3 - λ.minM3)
        >> filter(λ.span < 3)
        >> arrange(λ.span),
    )


@tables("df4")
def test_nested_bool(df4_x, df4_y):
    assert_result_equal(
        df4_x,
        df4_y,
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


@tables("df4")
def test_op_shift(df4_x, df4_y):
    assert_result_equal(
        df4_x,
        df4_y,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(
            shift1=t.col2.shift(1, arrange=[t.col4]),
            shift2=t.col4.shift(-2, arrange=[t.col4]),
        ),
    )


@tables("df4")
def test_op_row_number(df4_x, df4_y):
    assert_result_equal(
        df4_x,
        df4_y,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(
            row_number1=f.row_number(arrange=[λ.col4]),
            row_number2=f.row_number(arrange=[λ.col2, λ.col3]),
        ),
    )


@tables("df4")
def test_op_rank(df4_x, df4_y):
    assert_result_equal(
        df4_x,
        df4_y,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(
            rank1=t.col1.rank(),
            rank2=t.col2.rank(),
            rank3=t.col2.nulls_last().rank(),
            rank4=t.col5.nulls_first().rank(),
            rank5=(-t.col5.nulls_first()).rank(),
        ),
    )


@tables("df4")
def test_op_dense_rank(df4_x, df4_y):
    assert_result_equal(
        df4_x,
        df4_y,
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
