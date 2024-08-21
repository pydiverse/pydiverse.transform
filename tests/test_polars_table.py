from __future__ import annotations

import polars as pl
import pytest

from pydiverse.transform import λ
from pydiverse.transform.core import dtypes
from pydiverse.transform.core import functions as f
from pydiverse.transform.core.alignment import aligned, eval_aligned
from pydiverse.transform.core.dispatchers import Pipeable, verb
from pydiverse.transform.core.table import Table
from pydiverse.transform.core.verbs import *
from pydiverse.transform.eager.polars_table import PolarsEager
from pydiverse.transform.errors import AlignmentError
from tests.util import assert_equal

df1 = pl.DataFrame(
    {
        "col1": [1, 2, 3, 4],
        "col2": ["a", "b", "c", "d"],
    }
)

df2 = pl.DataFrame(
    {
        "col1": [1, 2, 2, 4, 5, 6],
        "col2": [2, 2, 0, 0, 2, None],
        "col3": [0.0, 0.1, 0.2, 0.3, 0.01, 0.02],
    }
)

df3 = pl.DataFrame(
    {
        "col1": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        "col2": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        "col3": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        "col4": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "col5": list("abcdefghijkl"),
    }
)

df4 = pl.DataFrame(
    {
        "col1": [None, 0, 0, 0, 0, None, 1, 1, 1, 2, 2, 2, 2],
        "col2": [0, 0, 1, 1, 0, 0, 1, None, 1, 0, 0, 1, 1],
        "col3": [None, None, None, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        "col4": [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "col5": list("abcdefghijkl") + [None],
    }
)

df_left = pl.DataFrame(
    {
        "a": [1, 2, 3, 4],
    }
)

df_right = pl.DataFrame(
    {
        "b": [0, 1, 2, 2],
        "c": [5, 6, 7, 8],
    }
)


@pytest.fixture(params=["numpy", "arrow"])
def dtype_backend(request):
    return request.param


@pytest.fixture
def tbl1():
    return Table(PolarsEager("df1", df1))


@pytest.fixture
def tbl2():
    return Table(PolarsEager("df2", df2))


@pytest.fixture
def tbl3():
    return Table(PolarsEager("df3", df3))


@pytest.fixture
def tbl4():
    return Table(PolarsEager("df4", df4.clone()))


@pytest.fixture
def tbl_left():
    return Table(PolarsEager("df_left", df_left.clone()))


@pytest.fixture
def tbl_right():
    return Table(PolarsEager("df_right", df_right.clone()))


def assert_not_inplace(tbl: Table[PolarsEager], operation: Pipeable):
    """
    Operations should not happen in-place. They should always return a new dataframe.
    """
    initial = tbl._impl.df.clone()
    tbl >> operation
    after = tbl._impl.df

    assert initial.equals(after)


class TestPolarsEager:
    def test_dtype(self, tbl1, tbl2):
        assert isinstance(tbl1.col1._.dtype, dtypes.Int)
        assert isinstance(tbl1.col2._.dtype, dtypes.String)

        assert isinstance(tbl2.col1._.dtype, dtypes.Int)
        assert isinstance(tbl2.col2._.dtype, dtypes.Int)
        assert isinstance(tbl2.col3._.dtype, dtypes.Float)

    def test_build_query(self, tbl1):
        assert (tbl1 >> build_query()) is None

    def test_export(self, tbl1):
        # TODO: test export to other backends
        assert_equal(tbl1, df1)

    def test_select(self, tbl1):
        assert_not_inplace(tbl1, select(tbl1.col1))
        assert_equal(tbl1 >> select(tbl1.col1), df1.select("col1"))
        assert_equal(tbl1 >> select(tbl1.col2), df1.select("col2"))
        assert_equal(tbl1 >> select(), df1.select())

    def test_mutate(self, tbl1):
        assert_not_inplace(tbl1, mutate(x=tbl1.col1))

        assert_equal(
            tbl1 >> mutate(col1times2=tbl1.col1 * 2),
            pl.DataFrame(
                {
                    "col1": [1, 2, 3, 4],
                    "col2": ["a", "b", "c", "d"],
                    "col1times2": [2, 4, 6, 8],
                }
            ),
        )

        assert_equal(
            tbl1 >> select() >> mutate(col1times2=tbl1.col1 * 2),
            pl.DataFrame(
                {
                    "col1times2": [2, 4, 6, 8],
                }
            ),
        )

        # Check proper column referencing
        t = tbl1 >> mutate(col2=tbl1.col1, col1=tbl1.col2) >> select()
        assert_equal(
            t >> mutate(x=t.col1, y=t.col2),
            tbl1 >> select() >> mutate(x=tbl1.col2, y=tbl1.col1),
        )
        assert_equal(
            t >> mutate(x=tbl1.col1, y=tbl1.col2),
            tbl1 >> select() >> mutate(x=tbl1.col1, y=tbl1.col2),
        )

    def test_join(self, tbl_left, tbl_right):
        assert_not_inplace(tbl_left, join(tbl_right, tbl_left.a == tbl_right.b, "left"))

        assert_equal(
            tbl_left
            >> join(tbl_right, tbl_left.a == tbl_right.b, "left")
            >> select(tbl_left.a, tbl_right.b),
            pl.DataFrame({"a": [1, 2, 2, 3, 4], "b_df_right": [1, 2, 2, None, None]}),
        )

        assert_equal(
            tbl_left
            >> join(tbl_right, tbl_left.a == tbl_right.b, "inner")
            >> select(tbl_left.a, tbl_right.b),
            pl.DataFrame({"a": [1, 2, 2], "b_df_right": [1, 2, 2]}),
        )

        assert_equal(
            tbl_left
            >> join(tbl_right, tbl_left.a == tbl_right.b, "outer")
            >> select(tbl_left.a, tbl_right.b),
            pl.DataFrame(
                {
                    "a": [None, 1, 2, 2, 3, 4],
                    "b_df_right": [0, 1, 2, 2, None, None],
                }
            ),
        )

    def test_filter(self, tbl1, tbl2):
        assert_not_inplace(tbl1, filter(tbl1.col1 == 3))

        # Simple filter expressions
        assert_equal(tbl1 >> filter(), df1)
        assert_equal(tbl1 >> filter(tbl1.col1 == tbl1.col1), df1)
        assert_equal(tbl1 >> filter(tbl1.col1 == 3), df1.filter(pl.col("col1") == 3))

        # More complex expressions
        assert_equal(
            tbl1 >> filter(tbl1.col1 // 2 == 1),
            pl.DataFrame({"col1": [2, 3], "col2": ["b", "c"]}),
        )

        assert_equal(
            tbl1 >> filter(1 < tbl1.col1) >> filter(tbl1.col1 < 4),
            df1.filter((1 < pl.col("col1")) & (pl.col("col1") < 4)),
        )

    def test_arrange(self, tbl2, tbl4):
        tbl4.col1.nulls_first()
        assert_not_inplace(tbl2, arrange(tbl2.col2))

        assert_equal(
            tbl2 >> arrange(tbl2.col3) >> select(tbl2.col3),
            df2.select(pl.col("col3")).sort("col3", descending=False),
        )

        assert_equal(
            tbl2 >> arrange(-tbl2.col3) >> select(tbl2.col3),
            df2.select(pl.col("col3")).sort("col3", descending=True),
        )

        assert_equal(
            tbl2 >> arrange(tbl2.col1, tbl2.col2),
            df2.sort(["col1", "col2"], descending=[False, False]),
        )

        assert_equal(
            tbl4
            >> arrange(
                tbl4.col1.nulls_first(),
                -tbl4.col2.nulls_first(),
                -tbl4.col5.nulls_first(),
            ),
            df4.sort(
                ["col1", "col2", "col5"],
                descending=[False, True, True],
                nulls_last=False,
            ),
        )

        assert_equal(
            tbl4
            >> arrange(
                tbl4.col1.nulls_last(),
                -tbl4.col2.nulls_last(),
                -tbl4.col5.nulls_last(),
            ),
            df4.sort(
                ["col1", "col2", "col5"],
                descending=[False, True, True],
                nulls_last=True,
            ),
        )

        assert_equal(
            tbl2 >> arrange(tbl2.col1, tbl2.col2),
            tbl2 >> arrange(tbl2.col2) >> arrange(tbl2.col1),
        )

        assert_equal(tbl2 >> arrange(--tbl2.col3), tbl2 >> arrange(tbl2.col3))  # noqa: B002

    def test_summarise(self, tbl3):
        assert_equal(
            tbl3 >> summarise(mean=tbl3.col1.mean(), max=tbl3.col4.max()),
            pl.DataFrame({"mean": [1.0], "max": [11]}),
            check_row_order=False,
        )

        assert_equal(
            tbl3 >> group_by(tbl3.col1) >> summarise(mean=tbl3.col4.mean()),
            pl.DataFrame({"col1": [0, 1, 2], "mean": [1.5, 5.5, 9.5]}),
            check_row_order=False,
        )

        assert_equal(
            tbl3 >> summarise(mean=tbl3.col4.mean()) >> mutate(mean_2x=λ.mean * 2),
            pl.DataFrame({"mean": [5.5], "mean_2x": [11.0]}),
            check_row_order=False,
        )

    def test_group_by(self, tbl3):
        # Grouping doesn't change the result
        assert_equal(tbl3 >> group_by(tbl3.col1), tbl3)
        assert_equal(
            tbl3 >> summarise(mean4=tbl3.col4.mean()) >> group_by(λ.mean4),
            tbl3 >> summarise(mean4=tbl3.col4.mean()),
            check_row_order=False,
        )

        # Groupings can be added
        assert_equal(
            tbl3
            >> group_by(tbl3.col1)
            >> group_by(tbl3.col2, add=True)
            >> summarise(mean3=tbl3.col3.mean(), mean4=tbl3.col4.mean()),
            tbl3
            >> group_by(tbl3.col1, tbl3.col2)
            >> summarise(mean3=tbl3.col3.mean(), mean4=tbl3.col4.mean()),
            check_row_order=False,
        )

        # Ungroup doesn't change the result
        assert_equal(
            tbl3
            >> group_by(tbl3.col1)
            >> summarise(mean4=tbl3.col4.mean())
            >> ungroup(),
            tbl3 >> group_by(tbl3.col1) >> summarise(mean4=tbl3.col4.mean()),
            check_row_order=False,
        )

    def test_alias(self, tbl1, tbl2):
        assert_not_inplace(tbl1, alias("tblxxx"))

        x = tbl2 >> alias("x")
        assert x._impl.name == "x"

        # Check that applying alias doesn't change the output
        a = (
            tbl1
            >> mutate(xyz=(tbl1.col1 * tbl1.col1) // 2)
            >> join(tbl2, tbl1.col1 == tbl2.col1, "left")
            >> mutate(col1=tbl1.col1 - λ.xyz)
        )
        b = a >> alias("b")

        assert_equal(a, b)

        # Self Join
        assert_equal(
            tbl2 >> join(x, tbl2.col1 == x.col1, "left"),
            df2.join(
                df2.rename(
                    mapping={"col1": "col1_x", "col2": "col2_x", "col3": "col3_x"}
                ),
                how="left",
                left_on="col1",
                right_on="col1_x",
                coalesce=False,
            ),
        )

    def test_window_functions(self, tbl3):
        # Everything else should stay the same
        assert_equal(
            tbl3 >> mutate(x=f.row_number(arrange=[-λ.col4])) >> select(*tbl3), df3
        )

        assert_equal(
            (
                tbl3
                >> group_by(λ.col2)
                >> select()
                >> mutate(x=f.row_number(arrange=[-λ.col4]))
            ),
            pl.DataFrame({"x": [6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1]}),
        )

    def test_slice_head(self, tbl3):
        @verb
        def slice_head_custom(tbl: Table, n: int, *, offset: int = 0):
            t = (
                tbl
                >> mutate(_n=f.row_number(arrange=[]))
                >> alias()
                >> filter((offset < λ._n) & (λ._n <= (n + offset)))
            )
            return t >> select(*[c for c in t if c._.name != "_n"])

        assert_equal(
            tbl3 >> slice_head(6),
            tbl3 >> slice_head_custom(6),
        )

        assert_equal(
            tbl3 >> slice_head(3, offset=2),
            tbl3 >> slice_head_custom(3, offset=2),
        )

        assert_equal(
            tbl3 >> slice_head(1, offset=100),
            tbl3 >> slice_head_custom(1, offset=100),
        )

    def test_case_expression(self, tbl3):
        assert_equal(
            (
                tbl3
                >> select()
                >> mutate(
                    col1=λ.col1.case(
                        (0, 1),
                        (1, 2),
                        (2, 3),
                        default=-1,
                    )
                )
            ),
            (df3.select("col1") + 1),
        )

        assert_equal(
            (
                tbl3
                >> select()
                >> mutate(
                    x=λ.col1.case(
                        (λ.col2, 1),
                        (λ.col3, 2),
                        default=0,
                    )
                )
            ),
            pl.DataFrame({"x": [1, 1, 0, 0, 0, 2, 1, 1, 0, 0, 2, 0]}),
        )

        assert_equal(
            (
                tbl3
                >> select()
                >> mutate(
                    x=f.case(
                        (λ.col1 == λ.col2, 1),
                        (λ.col1 == λ.col3, 2),
                        default=λ.col4,
                    )
                )
            ),
            pl.DataFrame({"x": [1, 1, 2, 3, 4, 2, 1, 1, 8, 9, 2, 11]}),
        )

    def test_lambda_column(self, tbl1, tbl2):
        # Select
        assert_equal(tbl1 >> select(λ.col1), tbl1 >> select(tbl1.col1))

        # Mutate
        assert_equal(
            tbl1 >> mutate(a=tbl1.col1 * 2) >> select() >> mutate(b=λ.a * 2),
            tbl1 >> select() >> mutate(b=tbl1.col1 * 4),
        )

        assert_equal(
            tbl1
            >> mutate(a=tbl1.col1 * 2)
            >> mutate(b=λ.a * 2, a=tbl1.col1)
            >> select(λ.b),
            tbl1 >> select() >> mutate(b=tbl1.col1 * 4),
        )

        # Join
        assert_equal(
            tbl1
            >> select()
            >> mutate(a=tbl1.col1)
            >> join(tbl2, λ.a == tbl2.col1, "left"),
            tbl1
            >> select()
            >> mutate(a=tbl1.col1)
            >> join(tbl2, tbl1.col1 == tbl2.col1, "left"),
        )

        # Join that also uses lambda for the right table
        assert_equal(
            tbl1
            >> select()
            >> mutate(a=tbl1.col1)
            >> join(tbl2, λ.a == λ.col1_df2, "left"),
            tbl1
            >> select()
            >> mutate(a=tbl1.col1)
            >> join(tbl2, tbl1.col1 == tbl2.col1, "left"),
        )

        # Filter
        assert_equal(
            tbl1 >> mutate(a=tbl1.col1 * 2) >> filter(λ.a % 2 == 0),
            tbl1 >> mutate(a=tbl1.col1 * 2) >> filter((tbl1.col1 * 2) % 2 == 0),
        )

        # Arrange
        assert_equal(
            tbl1 >> mutate(a=tbl1.col1 * 2) >> arrange(λ.a),
            tbl1 >> arrange(tbl1.col1) >> mutate(a=tbl1.col1 * 2),
        )

    def test_table_setitem(self, tbl_left, tbl_right):
        tl = tbl_left >> alias("df_left")
        tr = tbl_right >> alias("df_right")

        # Iterate over cols and modify
        for col in tl:
            tl[col] = (col * 2) % 3
        for col in tr:
            tr[col] = (col * 2) % 5

        # Check if it worked...
        assert_equal(
            (tl >> join(tr, λ.a == λ.b_df_right, "left")),
            (
                tbl_left
                >> mutate(a=(tbl_left.a * 2) % 3)
                >> join(
                    tbl_right
                    >> mutate(b=(tbl_right.b * 2) % 5, c=(tbl_right.c * 2) % 5),
                    λ.a == λ.b_df_right,
                    "left",
                )
            ),
        )

    def test_custom_verb(self, tbl1):
        @verb
        def double_col1(tbl):
            tbl[λ.col1] = λ.col1 * 2
            return tbl

        # Custom verb should not mutate input object
        assert_not_inplace(tbl1, double_col1())

        assert_equal(tbl1 >> double_col1(), tbl1 >> mutate(col1=λ.col1 * 2))


class TestPandasAligned:
    def test_eval_aligned(self, tbl1, tbl3, tbl_left, tbl_right):
        # No exception with correct length
        eval_aligned(tbl_left.a + tbl_left.a)
        eval_aligned(tbl_left.a + tbl_right.b)

        with pytest.raises(AlignmentError):
            eval_aligned(tbl1.col1 + tbl3.col1)

        # Test aggregate functions still work
        eval_aligned(tbl1.col1 + tbl3.col1.mean())

        # Test that `with_` argument gets enforced
        eval_aligned(tbl1.col1 + tbl1.col1, with_=tbl1)
        eval_aligned(tbl_left.a * 2, with_=tbl_left)
        eval_aligned(tbl_left.a * 2, with_=tbl_right)  # Same length
        eval_aligned(
            tbl1.col1.mean(), with_=tbl_left
        )  # Aggregate is aligned with everything

        with pytest.raises(AlignmentError):
            eval_aligned(tbl3.col1 * 2, with_=tbl1)

    def test_aligned_decorator(self, tbl1, tbl3, tbl_left, tbl_right):
        @aligned(with_="a")
        def f(a, b):
            return a + b

        f(tbl3.col1, tbl3.col2)
        f(tbl_left.a, tbl_right.b)

        with pytest.raises(AlignmentError):
            f(tbl1.col1, tbl3.col1)

        # Bad Alignment of return type
        @aligned(with_="a")
        def f(a, b):
            return a.mean() + b

        with pytest.raises(AlignmentError):
            f(tbl1.col1, tbl3.col1)

        # Invalid with_ argument
        with pytest.raises(ValueError):
            aligned(with_="x")(lambda a: 0)

    def test_col_addition(self, tbl_left, tbl_right):
        @aligned(with_="a")
        def f(a, b):
            return a + b

        assert_equal(
            tbl_left >> mutate(x=f(tbl_left.a, tbl_right.b)) >> select(λ.x),
            pl.DataFrame({"x": (df_left["a"] + df_right["b"])}),
        )

        with pytest.raises(AlignmentError):
            f(tbl_left.a, (tbl_right >> filter(λ.b == 2)).b)

        with pytest.raises(AlignmentError):
            x = f(tbl_left.a, tbl_right.b)
            tbl_left >> filter(λ.a <= 3) >> mutate(x=x)


class TestPrintAndRepr:
    def test_table_str(self, tbl1):
        # Table: df1, backend: PolarsEager
        #    col1 col2
        # 0     1    a
        # 1     2    b
        # 2     3    c
        # 3     4    d

        tbl_str = str(tbl1)

        assert "df1" in tbl_str
        assert "PolarsEager" in tbl_str
        assert str(df1) in tbl_str

    def test_table_repr_html(self, tbl1):
        # jupyter html
        assert "exception" not in tbl1._repr_html_()

    def test_col_str(self, tbl1):
        # Symbolic Expression: <df1.col1(int)>
        # dtype: int
        #
        # 0    1
        # 1    2
        # 2    3
        # 3    4
        # Name: df1_col1_XXXXXXXX, dtype: Int64

        col1_str = str(tbl1.col1)
        series = tbl1._impl.df[tbl1._impl.df_name_mapping[tbl1.col1._.uuid]]

        assert str(series) in col1_str
        assert "exception" not in col1_str

    def test_col_html_repr(self, tbl1):
        assert "exception" not in tbl1.col1._repr_html_()

    def test_expr_str(self, tbl1):
        expr_str = str(tbl1.col1 * 2)
        assert "exception" not in expr_str

    def test_expr_html_repr(self, tbl1):
        assert "exception" not in (tbl1.col1 * 2)._repr_html_()

    def test_lambda_str(self, tbl1):
        assert "exception" in str(λ.col)
        assert "exception" in str(λ.col1 + tbl1.col1)

    def test_eval_expr_str(self, tbl_left, tbl_right):
        valid = tbl_left.a + tbl_right.b
        invalid = tbl_left.a + (tbl_right >> filter(λ.b == 2)).b

        assert "exception" not in str(valid)
        assert "exception" in str(invalid)
