# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.transform as pdt
from pydiverse.transform.extended import *
from tests.fixtures.backend import skip_backends
from tests.util import assert_result_equal
from tests.util.filelock import lock


def test_union_basic(df3, df4):
    """Test basic union with only visible columns."""
    # Test pipe syntax: tbl1 >> union(tbl2)
    assert_result_equal(
        (df3, df4),
        lambda t, u: t >> union(u),
        check_row_order=False,
    )

    # Test direct call syntax: union(tbl1, tbl2)
    assert_result_equal(
        (df3, df4),
        lambda t, u: union(t, u),
        check_row_order=False,
    )


def test_union_all(df3, df4):
    """Test UNION ALL (keeping duplicates)."""
    # Test pipe syntax with distinct=False (keeps duplicates)
    assert_result_equal(
        (df3, df4),
        lambda t, u: t >> union(u, distinct=False),
        check_row_order=False,
    )

    # Test direct call syntax with distinct=False (keeps duplicates)
    assert_result_equal(
        (df3, df4),
        lambda t, u: union(t, u, distinct=False),
        check_row_order=False,
    )


@skip_backends("ibm_db2")
def test_union_distinct(df3, df4):
    """Test UNION (removing duplicates)."""
    # Test pipe syntax with distinct=True (default, removes duplicates)
    assert_result_equal(
        (df3, df4),
        lambda t, u: t >> union(u, distinct=True),
        check_row_order=False,
    )

    # Test direct call syntax with distinct=True (default, removes duplicates)
    assert_result_equal(
        (df3, df4),
        lambda t, u: union(t, u, distinct=True),
        check_row_order=False,
    )


def test_union_distinct2(df3, df4):
    """Test UNION (removing duplicates)."""
    # Test pipe syntax with distinct=True (default, removes duplicates)
    assert_result_equal(
        (df3, df4),
        lambda t, u: t >> drop(t.col7) >> union(u >> drop(u.col7), distinct=True),
        check_row_order=False,
    )

    # Test direct call syntax with distinct=True (default, removes duplicates)
    assert_result_equal(
        (df3, df4),
        lambda t, u: union(t >> drop(t.col7), u >> drop(u.col7), distinct=True),
        check_row_order=False,
    )


def test_union_with_select(df3, df4):
    """Test union with selected columns."""
    # Test with select on left table
    assert_result_equal(
        (df3, df4),
        lambda t, u: t >> select(t.col1) >> union(u >> select(u.col1)),
        check_row_order=False,
    )

    # Test with select on right table
    assert_result_equal(
        (df3, df4),
        lambda t, u: t >> select(t.col2) >> union(u >> select(u.col2)),
        check_row_order=False,
    )


def test_union_with_hidden_columns_left(df3, df4):
    """Test union when left table has hidden columns."""
    # Left table has hidden column, right table has all columns visible
    assert_result_equal(
        (df3, df4),
        lambda t, u: t
        >> mutate(hidden_col=t.col1 * 10)  # create a column
        >> select(t.col1, t.col2)  # hidden_col becomes hidden
        >> union(u >> select(u.col1, u.col2)),  # right has no hidden columns
        check_row_order=False,
    )


def test_union_with_hidden_columns_right(df3, df4):
    """Test union when right table has hidden columns that left table doesn't have."""
    # Right table has hidden column, left table has all columns visible
    # The hidden column in right should not appear in the union result
    assert_result_equal(
        (df3, df4),
        lambda t, u: t
        >> select(t.col1, t.col2)  # all visible
        >> union(u >> mutate(hidden_col=u.col1 * 10) >> select(u.col1, u.col2)),  # hidden_col becomes hidden in right
        check_row_order=False,
    )


def test_union_with_hidden_columns_partial_match(df3, df4):
    """Test union when both tables have hidden columns that partially match."""
    # Both tables have some hidden columns in common, some different
    # Only hidden columns that exist in BOTH tables should be preserved
    assert_result_equal(
        (df3, df4),
        lambda t, u: t
        >> mutate(shared_hidden=t.col1 * 2, left_only=t.col1 + 100)  # create columns
        >> select(t.col1, t.col2)  # shared_hidden and left_only become hidden
        >> union(
            u
            >> mutate(shared_hidden=u.col1 * 2, right_only=u.col1 + 200)
            >> select(u.col1, u.col2)  # shared_hidden and right_only become hidden
        ),
        check_row_order=False,
    )


def test_union_with_mutate_hidden(df3, df4):
    """Test union with mutated columns that become hidden."""
    # Create columns in mutate, then hide some
    assert_result_equal(
        (df3, df4),
        lambda t, u: t
        >> mutate(x=t.col1 * 2, y=t.col2 * 10)
        >> select(t.col1, C.x)  # y becomes hidden
        >> union(u >> mutate(x=u.col1 * 2, y=u.col2 * 10) >> select(u.col1, C.x)),
        check_row_order=False,
    )


def test_union_chained(df3, df4):
    """Test chaining multiple unions."""
    # Chain multiple unions together
    assert_result_equal(
        (df3, df4),
        lambda t, u: t >> union(u) >> union(t),
        check_row_order=False,
    )


@skip_backends("sqlite")  # sqlite only supports UNION for trivial queries
def test_union_after_operations(df3, df4):
    """Test union after other operations like filter and arrange."""
    assert_result_equal(
        (df3, df4),
        lambda t, u: t >> filter(t.col1 > 0) >> arrange(t.col1) >> union(u >> filter(u.col1 > 0) >> arrange(u.col1)),
        check_row_order=False,
    )


def test_union_error_different_columns(df3, df4):
    """Test that union raises error when columns don't match."""
    # Should raise ValueError when column names don't match
    assert_result_equal(
        (df3, df4),
        lambda t, u: t >> select(t.col1) >> union(u >> select(u.col2)),
        exception=ValueError,
    )


def test_union_error_different_backends():
    """Test that union raises error when backends don't match."""
    import polars as pl
    import sqlalchemy as sqa

    # Create tables with different backends
    polars_tbl = pdt.Table(pl.DataFrame({"a": [1, 2]}))
    file = "/tmp/transform/test2.sqlite"
    with lock(file):
        engine = sqa.create_engine("sqlite:///" + file)
        pl.DataFrame({"a": [1, 2]}).write_database("test", engine, if_table_exists="replace")
        sql_tbl = pdt.Table("test", pdt.SqlAlchemy(engine))

        # Should raise TypeError
        with pytest.raises(TypeError, match="cannot union two tables with different backends"):
            polars_tbl >> union(sql_tbl)


def test_union_error_grouped_table(df3, df4):
    """Test that union raises error when table is grouped."""
    # Should raise ValueError when trying to union a grouped table
    assert_result_equal(
        (df3, df4),
        lambda t, u: t >> group_by(t.col1) >> union(u),
        exception=ValueError,
    )

    assert_result_equal(
        (df3, df4),
        lambda t, u: t >> union(u >> group_by(u.col1)),
        exception=ValueError,
    )


def test_union_with_rename(df3, df4):
    """Test union after renaming columns."""
    assert_result_equal(
        (df3, df4),
        lambda t, u: t >> rename({"col1": "a", "col2": "b"}) >> union(u >> rename({"col1": "a", "col2": "b"})),
        check_row_order=False,
    )


def test_union_empty_tables():
    """Test union with empty tables."""
    import polars as pl

    empty1 = pdt.Table(pl.DataFrame({"a": [], "b": []}))
    empty2 = pdt.Table(pl.DataFrame({"a": [], "b": []}))

    result = empty1 >> union(empty2) >> export(pdt.Polars(lazy=False))
    assert len(result) == 0
    assert result.columns == ["a", "b"]


def test_union_different_column_order1(df3, df4):
    """Test union when columns are in different order (should reorder automatically)."""
    # Columns in different order should still work - union should handle reordering
    # Note: This tests that the backend correctly reorders columns to match
    assert_result_equal(
        (df3, df4),
        lambda t, u: t >> select(t.col1, t.col2) >> union(u >> select(u.col1, u.col2)),  # same columns, same order
        check_row_order=False,
    )


def test_union_different_column_order2(df3, df4):
    # Test that union handles column reordering correctly
    # The backend should reorder right table columns to match left order (by column name)
    assert_result_equal(
        (df3, df4),
        lambda t, u: t
        >> select(t.col1, t.col2)
        >> union(u >> select(u.col2, u.col1)),  # different order - should be reordered
        check_row_order=False,
    )


def test_union_different_column_order3(df3, df4):
    # Test that union handles column reordering correctly
    # The backend should not reorder right table columns since order matches based on name
    assert_result_equal(
        (df3, df4),
        lambda t, u: t
        >> select(t.col1, t.col2)
        >> union(u >> select(u.col2, u.col1) >> rename({u.col1: "col2", u.col2: "col1"})),
        check_row_order=False,
    )


def test_union_different_column_order4(df3, df4):
    # Test that union handles column reordering correctly
    # The backend should reorder right table columns to match left order
    assert_result_equal(
        (df3, df4),
        lambda t, u: t
        >> select(t.col1, t.col2)
        >> union(u >> select(u.col1, u.col2) >> rename({u.col1: "col2", u.col2: "col1"})),
        check_row_order=False,
    )
