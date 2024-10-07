from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

import pydiverse.transform as pdt
from pydiverse.transform.extended import *
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

df_dt = pl.DataFrame(
    {
        "dt1": [
            dt.datetime(1999, 10, 31, hour=15),
            dt.datetime(2013, 7, 14, second=13, minute=55),
        ],
        "dur1": [dt.timedelta(days=5), dt.timedelta(minutes=4, microseconds=197)],
        "dt2": [dt.datetime(1974, 12, 31), dt.datetime(2020, 2, 14, microsecond=43)],
        "d1": [dt.date(2002, 6, 28), dt.date(2003, 1, 1)],
    }
)


@pytest.fixture
def dtype_backend(request):
    return request.param


@pytest.fixture
def tbl1():
    return pdt.Table(df1, name="df1")


@pytest.fixture
def tbl2():
    return pdt.Table(df2, name="df2")


@pytest.fixture
def tbl3():
    return pdt.Table(df3, name="df3")


@pytest.fixture
def tbl4():
    return pdt.Table(df4, name="df4")


@pytest.fixture
def tbl_left():
    return pdt.Table(df_left, name="df_left")


@pytest.fixture
def tbl_right():
    return pdt.Table(df_right, name="df_right")


@pytest.fixture
def tbl_dt():
    return pdt.Table(df_dt)


class TestPolarsLazyImpl:
    def test_dtype(self, tbl1, tbl2):
        assert isinstance(tbl1.col1.dtype(), pdt.Int64)
        assert isinstance(tbl1.col2.dtype(), pdt.String)

        assert isinstance(tbl2.col1.dtype(), pdt.Int64)
        assert isinstance(tbl2.col2.dtype(), pdt.Int64)
        assert isinstance(tbl2.col3.dtype(), pdt.Float64)

        # test that column expression type errors are checked immediately
        with pytest.raises(TypeError):
            tbl1.col1 + tbl1.col2

        # here, transform should not be able to resolve the type and throw an error
        C.col1 + tbl1.col2

    def test_build_query(self, tbl1):
        assert (tbl1 >> build_query()) is None

    def test_export(self, tbl1):
        # TODO: test export to other backends
        assert_equal(tbl1, df1)

    def test_select(self, tbl1):
        assert_equal(tbl1 >> select(tbl1.col1), df1.select("col1"))
        assert_equal(tbl1 >> select(tbl1.col2), df1.select("col2"))
        assert_equal(tbl1 >> select(), df1.select())

    def test_mutate(self, tbl1):
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

        # # Check proper column referencing
        t = tbl1 >> mutate(col2=tbl1.col1, col1=tbl1.col2)
        assert_equal(
            t >> select() >> mutate(x=t.col1, y=t.col2),
            tbl1 >> select() >> mutate(x=tbl1.col2, y=tbl1.col1),
        )
        assert_equal(
            t >> select() >> mutate(x=tbl1.col1, y=tbl1.col2),
            tbl1 >> select() >> mutate(x=tbl1.col1, y=tbl1.col2),
        )

    def test_join(self, tbl_left, tbl_right):
        assert_equal(
            tbl_left
            >> join(tbl_right, tbl_left.a == tbl_right.b, "left")
            >> select(tbl_left.a, tbl_right.b),
            pl.DataFrame({"a": [1, 2, 2, 3, 4], "b_df_right": [1, 2, 2, None, None]}),
            check_row_order=False,
        )

        assert_equal(
            tbl_left
            >> join(tbl_right, tbl_left.a == tbl_right.b, "inner", suffix="")
            >> select(tbl_left.a, tbl_right.b),
            pl.DataFrame({"a": [1, 2, 2], "b": [1, 2, 2]}),
            check_row_order=False,
        )

        assert_equal(
            tbl_left
            >> join(tbl_right, tbl_left.a == tbl_right.b, "full", suffix="42")
            >> select(tbl_left.a, tbl_right.b),
            pl.DataFrame(
                {
                    "a": [None, 1, 2, 2, 3, 4],
                    "b42": [0, 1, 2, 2, None, None],
                }
            ),
            check_row_order=False,
        )

        # test self-join
        assert_equal(
            tbl_left
            >> inner_join(
                tbl_left2 := tbl_left >> alias(),
                tbl_left.a == tbl_left2.a,
            ),
            df_left.join(df_left, on="a", coalesce=False, suffix="_df_left"),
        )

        assert_equal(
            tbl_right
            >> inner_join(
                tbl_right2 := tbl_right >> alias(), tbl_right.b == tbl_right2.b
            )
            >> inner_join(
                tbl_right3 := tbl_right >> alias(), tbl_right.b == tbl_right3.b
            ),
            df_right.join(df_right, "b", suffix="_df_right", coalesce=False).join(
                df_right, "b", suffix="_df_right_1", coalesce=False
            ),
        )

    def test_filter(self, tbl1, tbl2):
        # Simple filter expressions
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
                tbl4.col2.nulls_first().descending(),
                tbl4.col5.nulls_first().descending(),
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
                tbl4.col2.descending().nulls_last(),
                tbl4.col5.descending().nulls_last(),
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

    def test_summarize(self, tbl3):
        assert_equal(
            tbl3 >> summarize(mean=tbl3.col1.mean(), max=tbl3.col4.max()),
            pl.DataFrame({"mean": [1.0], "max": [11]}),
            check_row_order=False,
        )

        assert_equal(
            tbl3 >> group_by(tbl3.col1) >> summarize(mean=tbl3.col4.mean()),
            pl.DataFrame({"col1": [0, 1, 2], "mean": [1.5, 5.5, 9.5]}),
            check_row_order=False,
        )

        assert_equal(
            tbl3 >> summarize(mean=tbl3.col4.mean()) >> mutate(mean_2x=C.mean * 2),
            pl.DataFrame({"mean": [5.5], "mean_2x": [11.0]}),
            check_row_order=False,
        )

    def test_group_by(self, tbl3):
        # Grouping doesn't change the result
        assert_equal(tbl3 >> group_by(tbl3.col1), tbl3)
        assert_equal(
            tbl3 >> summarize(mean4=tbl3.col4.mean()) >> group_by(C.mean4),
            tbl3 >> summarize(mean4=tbl3.col4.mean()),
            check_row_order=False,
        )

        # Groupings can be added
        assert_equal(
            tbl3
            >> group_by(tbl3.col1)
            >> group_by(tbl3.col2, add=True)
            >> summarize(mean3=tbl3.col3.mean(), mean4=tbl3.col4.mean()),
            tbl3
            >> group_by(tbl3.col1, tbl3.col2)
            >> summarize(mean3=tbl3.col3.mean(), mean4=tbl3.col4.mean()),
            check_row_order=False,
        )

        # Ungroup doesn't change the result
        assert_equal(
            tbl3
            >> group_by(tbl3.col1)
            >> summarize(mean4=tbl3.col4.mean())
            >> ungroup(),
            tbl3 >> group_by(tbl3.col1) >> summarize(mean4=tbl3.col4.mean()),
            check_row_order=False,
        )

    def test_alias(self, tbl1, tbl2):
        x = tbl2 >> alias("x")
        assert x._ast.name == "x"

        # Check that applying alias doesn't change the output
        a = (
            tbl1
            >> mutate(xyz=(tbl1.col1 * tbl1.col1) // 2)
            >> join(tbl2, tbl1.col1 == tbl2.col1, "left", suffix="_right")
            >> mutate(col1=tbl1.col1 - C.xyz)
        )
        b = a >> alias("b")

        assert_equal(a, b)

        # self join
        assert_equal(
            tbl2 >> join(x, tbl2.col1 == x.col1, "left", suffix="_right"),
            df2.join(
                df2,
                how="left",
                left_on="col1",
                right_on="col1",
                coalesce=False,
            ),
        )

    def test_window_functions(self, tbl3):
        # Everything else should stay the same
        assert_equal(
            tbl3 >> mutate(x=row_number(arrange=[-C.col4])) >> select(*tbl3),
            df3,
        )

        assert_equal(
            (
                tbl3
                >> group_by(C.col2)
                >> mutate(x=row_number(arrange=[-C.col4]))
                >> select(C.x)
            ),
            pl.DataFrame({"x": [6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1]}),
        )

        # group_by and partition_by should lead to the same result
        assert_equal(
            (
                tbl3
                >> group_by(C.col2)
                >> mutate(x=row_number(arrange=[-C.col4]))
                >> select(C.x)
            ),
            (
                tbl3
                >> mutate(x=row_number(arrange=[-C.col4], partition_by=[C.col2]))
                >> select(C.x)
            ),
        )

    def test_slice_head(self, tbl3):
        @pdt.verb
        def slice_head_custom(table: pdt.Table, n: int, *, offset: int = 0):
            t = (
                table
                >> mutate(_n=row_number(arrange=[]))
                >> alias()
                >> filter((offset < C._n) & (C._n <= (n + offset)))
            )
            return t >> select(*[C[col.name] for col in table if col.name != "_n"])

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
                >> mutate(
                    col1=when(C.col1 == 0)
                    .then(1)
                    .when(C.col1 == 1)
                    .then(2)
                    .when(C.col1 == 2)
                    .then(3)
                    .otherwise(-1)
                )
                >> select(C.col1)
            ),
            (df3.select("col1") + 1),
        )

        assert_equal(
            (
                tbl3
                >> mutate(
                    x=when(C.col1 == C.col2)
                    .then(1)
                    .when(C.col1 == C.col3)
                    .then(2)
                    .otherwise(C.col4)
                )
                >> select(C.x)
            ),
            pl.DataFrame({"x": [1, 1, 2, 3, 4, 2, 1, 1, 8, 9, 2, 11]}),
        )

    def test_lambda_column(self, tbl1, tbl2):
        # Select
        assert_equal(tbl1 >> select(C.col1), tbl1 >> select(tbl1.col1))

        # Mutate
        assert_equal(
            tbl1 >> mutate(a=tbl1.col1 * 2) >> mutate(b=C.a * 2) >> select(C.b),
            tbl1 >> select() >> mutate(b=tbl1.col1 * 4),
        )

        assert_equal(
            tbl1
            >> mutate(a=tbl1.col1 * 2)
            >> mutate(b=C.a * 2, a=tbl1.col1)
            >> select(C.b),
            tbl1 >> select() >> mutate(b=tbl1.col1 * 4),
        )

        # Join
        assert_equal(
            tbl1
            >> mutate(a=tbl1.col1)
            >> join(tbl2, C.a == tbl2.col1, "left")
            >> select(C.a, *tbl2),
            tbl1
            >> select()
            >> mutate(a=tbl1.col1)
            >> join(tbl2, tbl1.col1 == tbl2.col1, "left"),
        )

        # Filter
        assert_equal(
            tbl1 >> mutate(a=tbl1.col1 * 2) >> filter(C.a % 2 == 0),
            tbl1 >> mutate(a=tbl1.col1 * 2) >> filter((tbl1.col1 * 2) % 2 == 0),
        )

        # Arrange
        assert_equal(
            tbl1 >> mutate(a=tbl1.col1 * 2) >> arrange(C.a),
            tbl1 >> arrange(tbl1.col1) >> mutate(a=tbl1.col1 * 2),
        )

    def test_custom_verb(self, tbl1):
        @pdt.verb
        def double_col1(table):
            table >>= mutate(col1=C.col1 * 2)
            return table

        assert_equal(tbl1 >> double_col1(), tbl1 >> mutate(col1=C.col1 * 2))

    def test_null(self, tbl4):
        # see if we can compare to null
        assert_equal(
            tbl4 >> mutate(u=tbl4.col1 == tbl4.col3),
            df4.with_columns((pl.col("col1") == pl.col("col3")).alias("u")),
        )

        assert_equal(
            tbl4 >> mutate(u=tbl4.col3.is_null()),
            df4.with_columns(pl.col("col3").is_null().alias("u")),
        )

        # test fill_null
        assert_equal(
            tbl4 >> mutate(u=tbl4.col3.fill_null(tbl4.col2)),
            df4.with_columns(pl.col("col3").fill_null(pl.col("col2")).alias("u")),
        )
        assert_equal(
            tbl4 >> mutate(u=tbl4.col3.fill_null(tbl4.col2)),
            tbl4
            >> mutate(
                u=pdt.when(tbl4.col3.is_null()).then(tbl4.col2).otherwise(tbl4.col3)
            ),
        )

    def test_datetime(self, tbl_dt):
        assert_equal(
            tbl_dt
            >> mutate(
                u=(tbl_dt.dt1 - tbl_dt.dt2),
                v=tbl_dt.d1 - tbl_dt.d1,
                w=(tbl_dt.d1 - tbl_dt.dt2) + tbl_dt.dur1 + dt.timedelta(days=1),
            ),
            df_dt.with_columns(
                (pl.col("dt1") - pl.col("dt2")).alias("u"),
                pl.duration().alias("v"),
                (
                    (pl.col("d1") - pl.col("dt2"))
                    + pl.col("dur1")
                    + pl.lit(dt.timedelta(days=1))
                ).alias("w"),
            ),
        )

    def test_duckdb_execution(self, tbl2, tbl3):
        assert_equal(
            tbl3
            >> mutate(u=tbl3.col1 * 2)
            >> collect(DuckDb())
            >> mutate(v=tbl3.col3 + C.u),
            tbl3 >> mutate(u=tbl3.col1 * 2) >> mutate(v=C.col3 + C.u),
        )

        assert_equal(
            tbl3
            >> collect(DuckDb())
            >> left_join(
                tbl2 >> collect(DuckDb()), tbl3.col1 == tbl2.col1, suffix="_right"
            )
            >> mutate(v=tbl3.col3 + tbl2.col2)
            >> group_by(C.col2)
            >> summarize(y=C.col3_right.sum()),
            tbl3
            >> left_join(tbl2, C.col1 == C.col1_right, suffix="_right")
            >> mutate(v=C.col3 + C.col2_right)
            >> group_by(C.col2)
            >> summarize(y=C.col3_right.sum()),
            check_row_order=False,
        )


class TestPrintAndRepr:
    def test_table_str(self, tbl1):
        tbl_str = str(tbl1)

        assert "df1" in tbl_str
        assert "PolarsImpl" in tbl_str
        assert str(df1) in tbl_str
        tbl1 >> show()

    def test_table_repr_html(self, tbl1):
        # jupyter html
        assert "Failed" not in tbl1._repr_html_()

    def test_col_str(self, tbl1):
        col1_str = str(tbl1.col1)
        series = tbl1._ast.df.collect().get_column("col1")

        assert str(series) in col1_str
        assert "Failed" not in col1_str

    def test_col_html_repr(self, tbl1):
        assert "Failed" not in tbl1.col1._repr_html_()

    def test_expr_str(self, tbl1):
        expr_str = str(tbl1.col1 * 2)
        assert "Failed" not in expr_str

    def test_expr_html_repr(self, tbl1):
        assert "Failed" not in (tbl1.col1 * 2)._repr_html_()

    def test_lambda_str(self, tbl1):
        assert "Failed" not in str(C.col)
        assert "Failed" not in str(C.col1 + tbl1.col1)
