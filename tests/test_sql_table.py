from __future__ import annotations

import polars as pl
import pytest
import sqlalchemy as sqa

from pydiverse.transform import C
from pydiverse.transform._internal.backend.targets import Polars, SqlAlchemy
from pydiverse.transform._internal.pipe import functions as f
from pydiverse.transform._internal.pipe.table import Table
from pydiverse.transform._internal.pipe.verbs import *
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
    engine = sqa.create_engine("sqlite:///:memory:")
    # engine = sqa.create_engine("postgresql://sqa:Pydiverse23@127.0.0.1:6543")
    # engine = sqa.create_engine(
    #     "mssql+pyodbc://sqa:PydiQuant27@127.0.0.1:1433"
    #     "/master?driver=ODBC+Driver+18+for+SQL+Server&encrypt=no"
    # )

    df1.write_database("df1", engine, if_table_exists="replace")
    df2.write_database("df2", engine, if_table_exists="replace")
    df3.write_database("df3", engine, if_table_exists="replace")
    df4.write_database("df4", engine, if_table_exists="replace")
    df_left.write_database("df_left", engine, if_table_exists="replace")
    df_right.write_database("df_right", engine, if_table_exists="replace")
    return engine


@pytest.fixture
def tbl1(engine):
    return Table("df1", SqlAlchemy(engine))


@pytest.fixture
def tbl2(engine):
    return Table("df2", SqlAlchemy(engine))


@pytest.fixture
def tbl3(engine):
    return Table("df3", SqlAlchemy(engine))


@pytest.fixture
def tbl4(engine):
    return Table("df4", SqlAlchemy(engine))


@pytest.fixture
def tbl_left(engine):
    return Table("df_left", SqlAlchemy(engine))


@pytest.fixture
def tbl_right(engine):
    return Table("df_right", SqlAlchemy(engine))


class TestSqlTable:
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
        assert_equal(tbl1 >> export(Polars()), df1)

    def test_select(self, tbl1):
        assert_equal(tbl1 >> select(tbl1.col1), df1.select("col1"))
        assert_equal(tbl1 >> select(tbl1.col2), df1.select("col2"))

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
            tbl1 >> mutate(col1times2=tbl1.col1 * 2) >> select(C.col1times2),
            pl.DataFrame(
                {
                    "col1times2": [2, 4, 6, 8],
                }
            ),
        )

        # Check proper column referencing
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
            >> join(tbl_right, tbl_left.a == tbl_right.b, "left", suffix="")
            >> select(tbl_left.a, tbl_right.b),
            pl.DataFrame({"a": [1, 2, 2, 3, 4], "b": [1, 2, 2, None, None]}),
        )

        assert_equal(
            tbl_left
            >> join(tbl_right, tbl_left.a == tbl_right.b, "inner", suffix="")
            >> select(tbl_left.a, tbl_right.b),
            pl.DataFrame({"a": [1, 2, 2], "b": [1, 2, 2]}),
        )

        assert_equal(
            (
                tbl_left
                >> join(tbl_right, tbl_left.a == tbl_right.b, "full", suffix="_1729")
                >> select(tbl_left.a, tbl_right.b)
            ),
            pl.DataFrame(
                {
                    "a": [1, 2, 2, 3, 4, None],
                    "b_1729": [1, 2, 2, None, None, 0],
                }
            ),
            check_row_order=False,
        )

    def test_filter(self, tbl1):
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

    def test_summarize(self, tbl3):
        assert_equal(
            tbl3 >> summarize(mean=tbl3.col1.mean(), max=tbl3.col4.max()),
            pl.DataFrame({"mean": [1], "max": [11]}),
        )

        assert_equal(
            tbl3 >> group_by(tbl3.col1) >> summarize(mean=tbl3.col4.mean()),
            pl.DataFrame({"col1": [0, 1, 2], "mean": [1.5, 5.5, 9.5]}),
            check_row_order=False,
        )

        assert_equal(
            tbl3 >> summarize(mean=tbl3.col4.mean()) >> mutate(mean_2x=C.mean * 2),
            pl.DataFrame({"mean": [5.5], "mean_2x": [11]}),
        )

    def test_group_by(self, tbl3):
        # Grouping doesn't change the result
        assert_equal(tbl3 >> group_by(tbl3.col1), tbl3)
        assert_equal(
            tbl3 >> summarize(mean4=tbl3.col4.mean()) >> group_by(C.mean4),
            tbl3 >> summarize(mean4=tbl3.col4.mean()),
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
        )

        # Ungroup doesn't change the result
        assert_equal(
            tbl3
            >> group_by(tbl3.col1)
            >> summarize(mean4=tbl3.col4.mean())
            >> ungroup(),
            tbl3 >> group_by(tbl3.col1) >> summarize(mean4=tbl3.col4.mean()),
        )

    def test_alias(self, tbl1, tbl2):
        x = tbl2 >> alias("x")
        assert x._ast.name == "x"

        # Check that applying alias doesn't change the output
        a = (
            tbl1
            >> mutate(xyz=(tbl1.col1 * tbl1.col1) // 2)
            >> join(tbl2, tbl1.col1 == tbl2.col1, "left")
            >> mutate(col1=tbl1.col1 - C.xyz)
        )
        b = a >> alias("b")

        assert_equal(a, b)

        # Self Join
        self_join = (
            tbl2
            >> join(x, tbl2.col1 == x.col1, "left", suffix="42")
            >> alias("self_join")
        )

        self_join_expected = df2.join(
            df2,
            how="left",
            left_on="col1",
            right_on="col1",
            coalesce=False,
            suffix="42",
        )

        assert_equal(self_join, self_join_expected, check_row_order=False)

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

    def test_select_without_tbl_ref(self, tbl2):
        assert_equal(
            tbl2 >> summarize(count=f.count()),
            tbl2 >> summarize(count=f.count(tbl2.col1)),
        )

        assert_equal(
            tbl2 >> summarize(count=f.count()), pl.DataFrame({"count": [len(df2)]})
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

    def test_case_expression(self, tbl3):
        assert_equal(
            (
                tbl3
                >> mutate(
                    col1=f.when(C.col1 == 0)
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
                    x=f.when(C.col1 == C.col2)
                    .then(1)
                    .when(C.col1 == C.col3)
                    .then(2)
                    .otherwise(C.col4)
                )
                >> select(C.x)
            ),
            pl.DataFrame({"x": [1, 1, 2, 3, 4, 2, 1, 1, 8, 9, 2, 11]}),
        )
