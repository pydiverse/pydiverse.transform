from __future__ import annotations

import pytest

from pydiverse.transform import C
from pydiverse.transform.backend.table_impl import TableImpl
from pydiverse.transform.pipe.pipeable import (
    col_to_table,
    inverse_partial,
    verb,
)
from pydiverse.transform.pipe.verbs import (
    arrange,
    collect,
    filter,
    join,
    mutate,
    rename,
    select,
)


class TestTable:
    def test_getattr(self, tbl1):
        assert tbl1.col1._.name == "col1"
        assert tbl1.col2._.table == tbl1._impl

        with pytest.raises(AttributeError, match="colXXX"):
            _ = tbl1.colXXX

    def test_getitem(self, tbl1):
        assert tbl1.col1 == tbl1["col1"]
        assert tbl1.col2 == tbl1["col2"]

        assert tbl1.col2 == tbl1[tbl1.col2]
        assert tbl1.col2 == tbl1[C.col2]

    def test_iter(self, tbl1, tbl2):
        assert len(list(tbl1)) == len(list(tbl1._impl.selected_cols()))
        assert len(list(tbl2)) == len(list(tbl2._impl.selected_cols()))

        assert repr(list(tbl1)) == repr([tbl1.col1, tbl1.col2])
        assert repr(list(tbl2)) == repr([tbl2.col1, tbl2.col2, tbl2.col3])

        assert repr(list(tbl2 >> select(tbl2.col2))) == repr([tbl2.col2])

        joined = tbl1 >> join(tbl2 >> select(tbl2.col3), tbl1.col1 == tbl2.col2, "left")
        assert repr(list(joined)) == repr([tbl1.col1, tbl1.col2, joined.col3_mock2])

    def test_dir(self, tbl1):
        assert dir(tbl1) == ["col1", "col2"]
        assert dir(tbl1 >> mutate(x=tbl1.col1)) == ["col1", "col2", "x"]

    def test_contains(self, tbl1, tbl2):
        assert tbl1.col1 in tbl1
        assert tbl1.col2 in tbl1

        assert tbl1.col1 not in tbl2
        assert tbl1.col2 not in tbl2

        assert C.col1 in tbl1
        assert C.col2 in tbl1
        assert C.col3 not in tbl1

        assert C.col1 in tbl2
        assert C.col2 in tbl2
        assert C.col3 in tbl2
        assert C.col4 not in tbl2

        assert all(col in tbl1 for col in tbl1)
        assert all(col in tbl2 for col in tbl2)


class TestDispatchers:
    def test_inverse_partial(self):
        def x(a, b, c):
            return (a, b, c)

        assert inverse_partial(x, 1, 2)(0) == (0, 1, 2)
        assert inverse_partial(x, 1, c=2)(0, c=3) == (0, 1, 3)

    def test_pipeable(self):
        @verb
        def add(v1, v2):
            return v1 + v2

        @verb
        def subtract(v1, v2):
            return v1 - v2

        assert 1 >> add(2) == 3

        add_10 = add(5) >> add(5)
        assert 5 >> add_10 == 15

        assert 5 >> subtract(3) == 2
        assert 5 >> add_10 >> subtract(5) == 10

    def test_col_to_table(self, tbl1):
        assert col_to_table(15) == 15
        assert col_to_table(tbl1) == tbl1

        c1_tbl = col_to_table(tbl1.col1._)
        assert isinstance(c1_tbl, TableImpl)
        assert c1_tbl.available_cols == {tbl1.col1._.uuid}
        assert list(c1_tbl.named_cols.fwd) == ["col1"]


class TestBuiltinVerbs:
    def test_collect(self, tbl1):
        assert (tbl1 >> collect()) == ["col1", "col2"]

    def test_select(self, tbl1, tbl2):
        # Normal Selection
        assert (tbl1 >> select() >> collect()) == []
        assert (tbl1 >> select(tbl1.col1) >> collect()) == ["col1"]

        with pytest.raises(ValueError):
            tbl1 >> select(tbl2.col1)

        with pytest.raises(TypeError):
            tbl1 >> select(tbl1.col1 + tbl1.col1)

        # Selection with Ellipsis ...
        assert (tbl1 >> select(...) >> collect()) == ["col1", "col2"]
        assert (tbl1 >> select() >> select(...) >> collect()) == ["col1", "col2"]
        assert (tbl1 >> mutate(x=C.col1) >> select(...) >> collect()) == [
            "col1",
            "col2",
            "x",
        ]
        assert (
            tbl1
            >> select()
            >> mutate(x=C.col1, col1=C.col2)
            >> select(...)
            >> select(C.col1)
            >> select(...)
            >> collect()
        ) == ["col1", "col2", "x"]

        # Negative Selection
        assert (tbl1 >> select(-C.col1) >> collect()) == ["col2"]
        assert (tbl1 >> select(C.col1) >> select(-C.col1) >> collect()) == []
        assert (tbl1 >> mutate(x=C.col1) >> select(-C.x) >> collect()) == [
            "col1",
            "col2",
        ]
        assert (tbl1 >> mutate(x=C.col1) >> select(-C.col1) >> collect()) == [
            "col2",
            "x",
        ]
        assert (
            tbl1 >> mutate(x=C.col1, y=C.col2) >> select(-C.col1, -C.y) >> collect()
        ) == ["col2", "x"]

        with pytest.raises(ValueError):
            tbl1 >> select(-C.col1, C.col1)
        with pytest.raises(ValueError):
            tbl1 >> select(-C.col1, C.col2)
        with pytest.raises(ValueError):
            tbl1 >> select(C.col1, -C.col2)
        with pytest.raises(TypeError):
            tbl1 >> select(..., -C.col2)

        assert (tbl1 >> select(--C.col1) >> collect()) == ["col1"]  # noqa: B002
        assert (tbl1 >> select(+-+-C.col1) >> collect()) == ["col1"]

    def test_rename(self, tbl2):
        def assert_rename(name_map, expected):
            assert (tbl2 >> rename(name_map) >> collect()) == expected

        assert_rename({}, ["col1", "col2", "col3"])
        assert_rename({"col1": "col1"}, ["col1", "col2", "col3"])

        assert_rename({"col1": "1st col"}, ["1st col", "col2", "col3"])
        assert_rename({"col1": "A", "col2": "B", "col3": "C"}, ["A", "B", "C"])

        # Order of name_map shouldn't matter
        assert_rename({"col3": "C", "col2": "B", "col1": "A"}, ["A", "B", "C"])

        # Swap names
        assert_rename({"col1": "col2", "col2": "col1"}, ["col2", "col1", "col3"])

        # Rename + Select
        assert tbl2 >> select(C.col1) >> rename(
            {"col1": "A", "col2": "col1"}
        ) >> collect() == ["A"]

        assert tbl2 >> select(-C.col1) >> rename(
            {"col1": "A", "col2": "col1", "col3": "col2"}
        ) >> select(*tbl2) >> collect() == ["A", "col1", "col2"]

        assert tbl2 >> rename({"col1": "col2", "col2": "col1"}) >> select(
            tbl2.col1, tbl2.col2
        ) >> collect() == ["col2", "col1"]

        # Key and value must be strings
        with pytest.raises(TypeError):
            tbl2 >> rename({1: "name"})
        with pytest.raises(TypeError):
            tbl2 >> rename({"col1": 1})

        # Try to rename cols that don't exist
        with pytest.raises(KeyError):
            tbl2 >> rename({"NONE": "SOMETHING"})
        with pytest.raises(KeyError):
            tbl2 >> rename({"name": "name"})

        # Map multiple columns to the same name
        with pytest.raises(ValueError):
            tbl2 >> rename({"col1": "A", "col2": "A"})
        with pytest.raises(ValueError):
            tbl2 >> rename({"col1": "A"}) >> rename({"col2": "A"})

        # Rename column to one that already exists
        with pytest.raises(ValueError):
            tbl2 >> rename({"col1": "col2"})
        with pytest.raises(ValueError):
            tbl2 >> rename({"col1": "A"}) >> rename({"col2": "A"})
        with pytest.raises(ValueError):
            tbl2 >> mutate(A=C.col1) >> select(-C.A) >> rename({"col1": "A"})

    def test_mutate(self, tbl1, tbl2):
        assert (tbl1 >> mutate() >> collect()) == ["col1", "col2"]
        assert (tbl1 >> mutate(x=tbl1.col1) >> collect()) == ["col1", "col2", "x"]
        assert (tbl1 >> select() >> mutate(x=tbl1.col1) >> collect()) == ["x"]

        with pytest.raises(ValueError):
            tbl1 >> mutate(x=tbl2.col1)

        t = tbl1 >> mutate(x=tbl1.col1 + tbl1.col2)
        t >> mutate(y=t.x)

    def test_join(self, tbl1, tbl2):
        assert len(tbl1 >> join(tbl2, tbl1.col1 == tbl2.col1, "left") >> collect()) == 5

        assert (
            len(
                tbl1
                >> select()
                >> join(tbl2, tbl1.col1 == tbl2.col1, "left")
                >> collect()
            )
            == 3
        )
        assert (
            len(
                tbl1
                >> join(tbl2 >> select(), tbl1.col1 == tbl2.col1, "left")
                >> collect()
            )
            == 2
        )
        assert (
            len(
                tbl1
                >> select()
                >> join(tbl2 >> select(), tbl1.col1 == tbl2.col1, "left")
                >> select(tbl1.col1, tbl2.col2)
                >> collect()
            )
            == 2
        )

        with pytest.raises(ValueError):
            # self join without alias
            tbl1 >> join(tbl1, tbl1.col1 == tbl1.col1, "inner")

        # Test that joined columns are accessible
        t = tbl1 >> join(tbl2, tbl1.col1 == tbl2.col1, "left")
        t >> select(t.col1, t.col1_mock2)

        # Select Ellipsis
        assert (
            tbl1
            >> join(tbl2 >> select(), tbl1.col1 == tbl2.col1, "left")
            >> select(...)
            >> collect()
        ) == ["col1", "col2", "col1_mock2", "col2_mock2", "col3_mock2"]

    def test_filter(self, tbl1, tbl2):
        tbl1 >> filter()
        tbl1 >> filter(tbl1.col1)

        with pytest.raises(ValueError):
            tbl1 >> filter(tbl2.col1)

    def test_arrange(self, tbl1, tbl2):
        tbl1 >> arrange(tbl1.col1)
        tbl1 >> arrange(-tbl1.col1)
        tbl1 >> arrange(tbl1.col1, tbl1.col2)
        tbl1 >> arrange(tbl1.col1, -tbl1.col2)

        with pytest.raises(ValueError):
            tbl1 >> arrange(tbl2.col1)
        with pytest.raises(ValueError):
            tbl1 >> arrange(tbl1.col1, -tbl2.col1)
