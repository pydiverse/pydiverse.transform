from __future__ import annotations

import sqlite3

import polars as pl
import pytest
import sqlalchemy as sa

from pydiverse.transform import λ
from pydiverse.transform.core import functions as f
from pydiverse.transform.core.alignment import aligned, eval_aligned
from pydiverse.transform.core.table import Table
from pydiverse.transform.core.verbs import *
from pydiverse.transform.errors import AlignmentError
from pydiverse.transform.sql.sql_table import SQLTableImpl
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


@pytest.fixture
def engine():
    engine = sa.create_engine("sqlite:///:memory:")
    df1.write_database("df1", engine, if_table_exists="replace")
    df2.write_database("df2", engine, if_table_exists="replace")
    df3.write_database("df3", engine, if_table_exists="replace")
    df4.write_database("df4", engine, if_table_exists="replace")
    df_left.write_database("df_left", engine, if_table_exists="replace")
    df_right.write_database("df_right", engine, if_table_exists="replace")
    return engine


@pytest.fixture
def tbl1(engine):
    return Table(SQLTableImpl(engine, "df1"))


@pytest.fixture
def tbl2(engine):
    return Table(SQLTableImpl(engine, "df2"))


@pytest.fixture
def tbl3(engine):
    return Table(SQLTableImpl(engine, "df3"))


@pytest.fixture
def tbl4(engine):
    return Table(SQLTableImpl(engine, "df4"))


@pytest.fixture
def tbl_left(engine):
    return Table(SQLTableImpl(engine, "df_left"))


@pytest.fixture
def tbl_right(engine):
    return Table(SQLTableImpl(engine, "df_right"))


class TestSQLTable:
    def test_build_query(self, tbl1):
        query_str = tbl1 >> build_query()
        expected_out = "SELECT df1.col1 AS col1, df1.col2 AS col2 FROM df1"
        assert query_str.lower().split() == expected_out.lower().split()

    def test_show_query(self, tbl1, capfd):
        tbl1 >> show_query()
        out = capfd.readouterr().out
        expected_out = "SELECT df1.col1 AS col1, df1.col2 AS col2 FROM df1"

        assert out.lower().split() == expected_out.lower().split()

        # Verify that it is chainable
        tbl1 >> show_query() >> collect()

    def test_export(self, tbl1):
        assert_equal(tbl1 >> export(), df1)

    def test_select(self, tbl1, tbl2):
        assert_equal(tbl1 >> select(tbl1.col1), df1[["col1"]])
        assert_equal(tbl1 >> select(tbl1.col2), df1[["col2"]])

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

        if sqlite3.sqlite_version_info >= (3, 39, 0):
            assert_equal(
                (
                    tbl_left
                    >> join(tbl_right, tbl_left.a == tbl_right.b, "outer")
                    >> select(tbl_left.a, tbl_right.b)
                ),
                pl.DataFrame(
                    {
                        "a": [1.0, 2.0, 2.0, 3.0, 4.0, None],
                        "b_df_right": [1.0, 2.0, 2.0, None, None, 0.0],
                    }
                ),
            )

    def test_filter(self, tbl1, tbl2):
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

    def test_arrange(self, tbl2):
        assert_equal(
            tbl2 >> arrange(tbl2.col3) >> select(tbl2.col3),
            df2.select("col3").sort("col3", descending=False),
        )

        assert_equal(
            tbl2 >> arrange(-tbl2.col3) >> select(tbl2.col3),
            df2.select("col3").sort("col3", descending=True),
        )

        assert_equal(
            tbl2 >> arrange(tbl2.col1, tbl2.col2),
            df2.sort(["col1", "col2"], descending=[False, False]),
        )

        assert_equal(
            tbl2 >> arrange(tbl2.col1, tbl2.col2),
            tbl2 >> arrange(tbl2.col2) >> arrange(tbl2.col1),
        )

        assert_equal(tbl2 >> arrange(--tbl2.col3), tbl2 >> arrange(tbl2.col3))  # noqa: B002

    def test_summarise(self, tbl3):
        assert_equal(
            tbl3 >> summarise(mean=tbl3.col1.mean(), max=tbl3.col4.max()),
            pl.DataFrame({"mean": [1], "max": [11]}),
        )

        assert_equal(
            tbl3 >> group_by(tbl3.col1) >> summarise(mean=tbl3.col4.mean()),
            pl.DataFrame({"col1": [0, 1, 2], "mean": [1.5, 5.5, 9.5]}),
        )

        assert_equal(
            tbl3 >> summarise(mean=tbl3.col4.mean()) >> mutate(mean_2x=λ.mean * 2),
            pl.DataFrame({"mean": [5.5], "mean_2x": [11]}),
        )

    def test_group_by(self, tbl3):
        # Grouping doesn't change the result
        assert_equal(tbl3 >> group_by(tbl3.col1), tbl3)
        assert_equal(
            tbl3 >> summarise(mean4=tbl3.col4.mean()) >> group_by(λ.mean4),
            tbl3 >> summarise(mean4=tbl3.col4.mean()),
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
        )

        # Ungroup doesn't change the result
        assert_equal(
            tbl3
            >> group_by(tbl3.col1)
            >> summarise(mean4=tbl3.col4.mean())
            >> ungroup(),
            tbl3 >> group_by(tbl3.col1) >> summarise(mean4=tbl3.col4.mean()),
        )

    def test_alias(self, tbl1, tbl2):
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
        self_join = tbl2 >> join(x, tbl2.col1 == x.col1, "left") >> alias("self_join")
        self_join >>= arrange(*self_join)

        self_join_expected = df2.join(
            df2.rename(mapping={"col1": "col1_x", "col2": "col2_x", "col3": "col3_x"}),
            how="left",
            left_on="col1",
            right_on="col1_x",
            coalesce=False,
        )
        self_join_expected = self_join_expected.sort(
            by=[col._.name for col in self_join]
        )

        assert_equal(self_join, self_join_expected)

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
            >> mutate(a=tbl1.col1 * 2)
            >> join(tbl2, λ.a == tbl2.col1, "left"),
            tbl1
            >> select()
            >> mutate(a=tbl1.col1 * 2)
            >> join(tbl2, tbl1.col1 * 2 == tbl2.col1, "left"),
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

    def test_select_without_tbl_ref(self, tbl2):
        assert_equal(
            tbl2 >> summarise(count=f.count()),
            tbl2 >> summarise(count=f.count(tbl2.col1)),
        )

        assert_equal(
            tbl2 >> summarise(count=f.count()), pl.DataFrame({"count": [len(df2)]})
        )

    def test_null_comparison(self, tbl4):
        assert_equal(
            tbl4 >> mutate(u=tbl4.col1 == tbl4.col3),
            df4.with_columns((pl.col("col1") == pl.col("col3")).alias("u")),
        )

        assert_equal(
            tbl4 >> mutate(u=tbl4.col3.is_null()),
            df4.with_columns(pl.col("col3").is_null().alias("u")),
        )


class TestSQLAligned:
    def test_eval_aligned(self, tbl1, tbl3, tbl_left, tbl_right):
        # Columns must be from same table
        eval_aligned(tbl_left.a + tbl_left.a)
        eval_aligned(tbl3.col1 + tbl3.col2)

        # Derived columns are also OK
        tbl1_mutate = tbl1 >> mutate(x=tbl1.col1 * 2)
        eval_aligned(tbl1.col1 + tbl1_mutate.x)

        with pytest.raises(AlignmentError):
            eval_aligned(tbl1.col1 + tbl3.col1)
        with pytest.raises(AlignmentError):
            eval_aligned(tbl_left.a + tbl_right.b)
        with pytest.raises(AlignmentError):
            eval_aligned(tbl1.col1 + tbl3.col1.mean())
        with pytest.raises(AlignmentError):
            tbl1_joined = tbl1 >> join(tbl3, tbl1.col1 == tbl3.col1, how="left")
            eval_aligned(tbl1.col1 + tbl1_joined.col1)

        # Test that `with_` argument gets enforced
        eval_aligned(tbl1.col1 + tbl1.col1, with_=tbl1)
        eval_aligned(tbl_left.a * 2, with_=tbl_left)
        eval_aligned(tbl1.col1, with_=tbl1_mutate)

        with pytest.raises(AlignmentError):
            eval_aligned(tbl1.col1.mean(), with_=tbl_left)

        with pytest.raises(AlignmentError):
            eval_aligned(tbl3.col1 * 2, with_=tbl1)

        with pytest.raises(AlignmentError):
            eval_aligned(tbl_left.a, with_=tbl_right)

    def test_aligned_decorator(self, tbl1, tbl3, tbl_left, tbl_right):
        @aligned(with_="a")
        def f(a, b):
            return a + b

        f(tbl3.col1, tbl3.col2)
        f(tbl_right.b, tbl_right.c)

        with pytest.raises(AlignmentError):
            f(tbl1.col1, tbl3.col1)

        with pytest.raises(AlignmentError):
            f(tbl_left.a, tbl_right.b)

        # Check with_ parameter gets enforced
        @aligned(with_="a")
        def f(a, b):
            return b

        f(tbl1.col1, tbl1.col2)
        with pytest.raises(AlignmentError):
            f(tbl1.col1, tbl3.col1)

        # Invalid with_ argument
        with pytest.raises(ValueError):
            aligned(with_="x")(lambda a: 0)

    def test_col_addition(self, tbl3):
        @aligned(with_="a")
        def f(a, b):
            return a + b

        assert_equal(
            tbl3 >> mutate(x=f(tbl3.col1, tbl3.col2)) >> select(λ.x),
            pl.DataFrame({"x": [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3]}),
        )

        # Test if it also works with derived tables
        tbl3_mutate = tbl3 >> mutate(x=tbl3.col1 * 2)
        tbl3 >> mutate(x=f(tbl3_mutate.col1, tbl3_mutate.x))

        with pytest.raises(AlignmentError):
            tbl3 >> arrange(λ.col1) >> mutate(x=f(tbl3.col1, tbl3.col2))

        with pytest.raises(AlignmentError):
            tbl3 >> filter(λ.col1 == 1) >> mutate(x=f(tbl3.col1, tbl3.col2))
