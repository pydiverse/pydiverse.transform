from __future__ import annotations

from pydiverse.transform import C
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
from tests.util import assert_result_equal, full_sort


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


def test_partition_by_argument(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(
            u=t.col1.min(partition_by=[t.col3]),
            v=t.col4.sum(partition_by=[t.col2]),
            w=f.rank(arrange=[-t.col5, t.col4], partition_by=[t.col2]),
            x=f.row_number(
                arrange=[t.col4.nulls_last()], partition_by=[t.col1, t.col2]
            ),
        ),
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
        >> mutate(range_mean=C.range.mean()),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(x=C.col4.max())
        >> mutate(y=C.x.min() * 1)
        >> mutate(z=C.y.mean())
        >> mutate(w=C.x / C.y),
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
        >> filter(C.mean3 <= 2.0),
    )


# def test_filter_argument(df3, df4):
#     assert_result_equal(
#         df4, lambda t: t >> mutate(u=t.col2.mean(filter=~t.col2.is_null()))
#     )

#     assert_result_equal(
#         df4, lambda t: t >> mutate(u=t.col2.mean(filter=~(t.col4 % 3 == 0)))
#     )

#     assert_result_equal(
#         df3, lambda t: t >> mutate(u=t.col4.sum(partition_by=[t.col2]))
#     )

#     assert_result_equal(
#         df4,
#         lambda t: t
#         >> mutate(
#             u=t.col1.min(filter=(~t.col1.is_null()), partition_by=[t.col3]),
#             v=t.col4.max(filter=~t.col4.is_null(), partition_by=[t.col1]),
#         ),
#     )

#     assert_result_equal(
#         df4, lambda t: t >> mutate(u=t.col3.min(filter=t.col3.is_null()))
#     )


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


def test_summarise(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(range=t.col4.max() - t.col4.min())
        >> summarise(mean_range=C.range.mean()),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(range=t.col4.max() - t.col4.min())
        >> mutate(mean_range=C.range.mean()),
    )


def test_intermediate_select(df3):
    # Check that subqueries happen transparently
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(x=t.col4.mean())
        >> select()
        >> mutate(y=C.x.min())
        >> select()
        >> mutate(z=(C.x - C.y).mean()),
    )


def test_arrange_argument(df3):
    # Grouped
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(x=C.col4.shift(1, arrange=[-C.col3]))
        >> full_sort()
        >> select(C.x),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col2)
        >> mutate(x=f.row_number(arrange=[-C.col4]))
        >> full_sort()
        >> select(C.x),
    )

    # Ungrouped
    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(x=C.col4.shift(1, arrange=[-C.col3]))
        >> full_sort()
        >> select(C.x),
    )

    assert_result_equal(
        df3,
        lambda t: t
        >> mutate(x=f.row_number(arrange=[-C.col4]))
        >> full_sort()
        >> select(C.x),
    )


def test_complex(df3):
    # Window function before summarise
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> mutate(mean3=t.col3.mean(), rn=f.row_number(arrange=[C.col1, C.col2]))
        >> filter(C.mean3 > C.rn)
        >> summarise(meta_mean=C.mean3.mean())
        >> filter(t.col1 >= C.meta_mean)
        >> filter(t.col1 != 1)
        >> arrange(C.meta_mean),
    )

    # Window function after summarise
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1, t.col2)
        >> summarise(mean3=t.col3.mean())
        >> mutate(minM3=C.mean3.min(), maxM3=C.mean3.max())
        >> mutate(span=C.maxM3 - C.minM3)
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
            xshift=C.x.shift(1, arrange=[t.col4]),
            yshift=C.y.shift(-1, arrange=[t.col4]),
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
            shift1=t.col2.shift(1, arrange=[t.col4]),
            shift2=t.col4.shift(-2, 0, arrange=[t.col4]),
            shift3=t.col4.shift(0, arrange=[t.col4]),
            u=C.col1.shift(1, 0, arrange=[t.col4]),
        ),
    )

    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            u=t.col1.shift(1, 0, arrange=[t.col2, t.col4]),
            v=t.col1.shift(2, 1, arrange=[-t.col4.nulls_first()]),
        ),
    )


def test_op_row_number(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(
            row_number1=f.row_number(arrange=[-C.col4.nulls_last()]),
            row_number2=f.row_number(arrange=[C.col2, C.col3, t.col4]),
        ),
    )

    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            u=f.row_number(arrange=[-C.col4.nulls_last()]),
            v=f.row_number(arrange=[-t.col3, t.col4]),
        ),
    )


def test_op_rank(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(
            rank1=f.rank(arrange=[t.col1]),
            rank2=f.rank(arrange=[t.col2]),
            rank3=f.rank(arrange=[t.col2.nulls_last()]),
            rank4=f.rank(arrange=[t.col5.nulls_first()]),
            rank5=f.rank(arrange=[-t.col5.nulls_first()]),
            rank_expr=f.rank(arrange=[t.col3 - t.col2]),
        ),
    )


def test_op_dense_rank(df3):
    assert_result_equal(
        df3,
        lambda t: t
        >> group_by(t.col1)
        >> mutate(
            rank1=f.dense_rank(arrange=[t.col5.nulls_first()]),
            rank2=f.dense_rank(arrange=[t.col2]),
            rank3=f.dense_rank(arrange=[t.col2.nulls_last()]),
        )
        >> ungroup(),
        # TODO: activate these once SQL partition_by= is implemented
        # >> mutate(
        #    rank4=f.dense_rank(arrange=[t.col4.nulls_first()], partition_by=[t.col2]),
        #    rank5=f.dense_rank(arrange=[-t.col5.nulls_first()], partition_by=[t.col2]),
        # ),
    )
