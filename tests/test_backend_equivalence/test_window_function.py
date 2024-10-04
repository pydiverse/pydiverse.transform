from __future__ import annotations

import pydiverse.transform as pdt
from pydiverse.transform._internal.errors import FunctionTypeError, SubqueryError
from pydiverse.transform.extended import *
from tests.util import assert_result_equal


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


def test_partition_by_argument(df3, df4):
    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(
            u=t.col1.min(partition_by=t.col3),
            v=t.col4.sum(partition_by=t.col2),
            w=pdt.rank(
                arrange=[t.col5.descending().nulls_last(), t.col4.nulls_first()],
                partition_by=[t.col2],
            ),
            x=pdt.row_number(
                arrange=[t.col4.nulls_last()], partition_by=[t.col1, t.col2]
            ),
        ),
    )

    assert_result_equal(
        (df3, df4),
        lambda t, u: t
        >> join(u, t.col1 == u.col3, how="left")
        >> group_by(t.col2)
        >> mutate(y=(u.col3 + t.col1).max(partition_by=(col for col in t))),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(
            u=t.col3.sum(),
            v=t.col2.sum(partition_by=[t.col2]),
        ),
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
        >> mutate(span=C.max - C.min),
    )


def test_nested(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(range=t.col4.max() - 10)
        >> ungroup()
        >> alias()
        >> mutate(range_mean=C.range.mean()),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(x=C.col4.max())
        >> mutate(y=C.x.min() * 1)
        >> mutate(z=C.y.mean())
        >> mutate(w=C.x / C.y),
        may_throw=True,
        exception=SubqueryError,
    )

    assert_result_equal(
        df3,
        lambda t: t >> mutate(x=(C.col4.max().min() + C.col2.mean()).max()),
        exception=FunctionTypeError,
        may_throw=True,
    )


def test_filter(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(mean3=t.col3.mean())
        >> alias()
        >> filter(C.mean3 <= 2.0),
    )


def test_filter_argument(df3, df4):
    assert_result_equal(
        df4, lambda t: t >> mutate(u=t.col2.mean(filter=~t.col2.is_null()))
    )

    assert_result_equal(
        df4, lambda t: t >> mutate(u=t.col2.mean(filter=~(t.col4 % 3 == 0)))
    )

    assert_result_equal(df3, lambda t: t >> mutate(u=t.col4.sum(partition_by=t.col2)))

    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            u=t.col1.min(filter=(~t.col1.is_null()), partition_by=[t.col3]),
            v=t.col4.max(filter=~t.col4.is_null(), partition_by=[t.col1]),
        ),
    )

    assert_result_equal(
        df4, lambda t: t >> mutate(u=t.col3.min(filter=t.col3.is_null()))
    )


def test_arrange(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(mean3=t.col3.mean())
        >> arrange(C.mean3),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> arrange(-t.col4)
        >> group_by(t.col1, t.col2)
        >> mutate(mean3=t.col3.mean())
        >> arrange(C.mean3),
    )


def test_summarize(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(range=t.col4.max() - t.col4.min())
        >> alias()
        >> summarize(mean_range=C.range.mean()),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarize(range=t.col4.max() - t.col4.min())
        >> alias()
        >> mutate(mean_range=C.range.mean()),
    )


def test_intermediate_select(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(x=t.col4.mean())
        >> alias()
        >> mutate(y=C.x.min())
        # TODO: technically, we could remove some window functions here and prevent
        # subqueries
        >> alias()
        >> mutate(z=(C.x - C.y).mean())
        >> select(C.z),
    )


def test_arrange_argument(df3):
    # Grouped
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(x=C.col4.shift(1, arrange=C.col3.nulls_last()))
        >> select(C.x),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col2)
        >> mutate(x=pdt.row_number(arrange=C.col4.descending()))
        >> select(C.x),
    )

    # Ungrouped
    assert_result_equal(
        df3,
        lambda t: t >> mutate(x=C.col4.shift(1, arrange=[-C.col3])) >> select(C.x),
    )

    assert_result_equal(
        df3,
        lambda t: t >> mutate(x=pdt.row_number(arrange=[-C.col4])) >> select(C.x),
    )


def test_complex(df3):
    # Window function before summarize
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(mean3=t.col3.mean(), rn=pdt.row_number(arrange=[C.col1, C.col2]))
        >> alias()
        >> filter(C.mean3 > C.rn)
        >> alias()
        >> summarize(meta_mean=C.mean3.mean())
        >> filter(C.col1 >= C.meta_mean)
        >> filter(C.col1 != 1)
        >> arrange(C.meta_mean),
    )

    # Window function after summarize
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarize(mean3=t.col3.mean())
        >> alias()
        >> mutate(minM3=C.mean3.min(), maxM3=C.mean3.max())
        >> mutate(span=C.maxM3 - C.minM3)
        >> alias()
        >> filter(C.span < 3)
        >> arrange(C.span),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarize(mean3=t.col3.mean(), u=t.col4.max())
        >> group_by(C.u)
        >> alias()
        >> mutate(minM3=C.mean3.min(), maxM3=C.mean3.max())
        >> mutate(span=C.maxM3 - C.minM3)
        >> alias()
        >> filter(C.span < 3)
        >> arrange(C.span),
    )


def test_nested_bool(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(x=t.col1 <= t.col2, y=(t.col3 * 4) >= C.col4)
        >> mutate(
            xshift=C.x.shift(1, arrange=[t.col4.nulls_last()]),
            yshift=C.y.shift(-1, arrange=[t.col4.nulls_first()]),
        )
        >> mutate(xAndY=C.x & C.y, xAndYshifted=C.xshift & C.yshift),
    )


# Test specific operations


def test_op_shift(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(
            shift1=t.col2.shift(1, arrange=[t.col4.nulls_first()]),
            shift2=t.col4.shift(-2, 0, arrange=[t.col4.nulls_last()]),
            shift3=t.col4.shift(0, arrange=[t.col4.nulls_first()]),
            u=C.col1.shift(1, 0, arrange=[t.col4.nulls_last()]),
        ),
    )

    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            u=t.col1.shift(1, 0, arrange=[t.col2.nulls_last(), t.col4.nulls_first()]),
            v=t.col1.shift(2, 1, arrange=[t.col4.descending().nulls_first()]),
        ),
    )


def test_op_row_number(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(
            row_number1=pdt.row_number(arrange=[C.col4.descending().nulls_last()]),
            row_number2=pdt.row_number(
                arrange=[C.col2.nulls_last(), C.col3.nulls_first(), t.col4.nulls_last()]
            ),
        ),
    )

    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            u=pdt.row_number(arrange=[C.col4.descending().nulls_last()]),
            v=pdt.row_number(
                arrange=[t.col3.descending().nulls_first(), t.col4.nulls_first()]
            ),
        ),
    )


def test_op_rank(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(
            rank1=pdt.rank(arrange=[t.col1.nulls_last()]),
            rank2=pdt.rank(arrange=[t.col2.nulls_first()]),
            rank3=pdt.rank(arrange=[t.col2.nulls_last()]),
            rank4=pdt.rank(arrange=[t.col5.nulls_first()]),
            rank5=pdt.rank(arrange=[t.col5.descending().nulls_first()]),
            rank_expr=pdt.rank(arrange=[(t.col3 - t.col2).nulls_last()]),
        ),
    )


def test_op_dense_rank(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(
            rank1=pdt.dense_rank(arrange=[t.col5.nulls_first()]),
            rank2=pdt.dense_rank(arrange=[t.col2.nulls_last()]),
            rank3=pdt.dense_rank(arrange=[t.col2.nulls_last()]),
        )
        >> ungroup()
        >> mutate(
            rank4=pdt.dense_rank(arrange=[t.col4.nulls_first()], partition_by=[t.col2]),
            rank5=pdt.dense_rank(
                arrange=[t.col5.descending().nulls_first()],
                partition_by=[t.col2],
            ),
        ),
    )
