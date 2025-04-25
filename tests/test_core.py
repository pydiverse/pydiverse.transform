from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

import pydiverse.transform as pdt
from pydiverse.transform._internal.errors import ColumnNotFoundError
from pydiverse.transform._internal.pipe.pipeable import Pipeable, inverse_partial
from pydiverse.transform.common import *


@pytest.fixture
def tbl1():
    return pdt.Table(pl.DataFrame({"col1": [0.1], "col2": [3.14]}))


@pytest.fixture
def tbl2():
    return pdt.Table(pl.DataFrame({"col1": [1], "col2": [2], "col3": [3]}))


class TestTable:
    def test_getattr(self, tbl1):
        assert tbl1.col1.name == "col1"
        assert tbl1.col2._ast == tbl1._ast

        with pytest.raises(ColumnNotFoundError, match="colXXX"):
            _ = tbl1.colXXX

    def test_getitem(self, tbl1):
        assert tbl1.col1._uuid == tbl1["col1"]._uuid
        assert tbl1.col2._uuid is tbl1["col2"]._uuid

    def test_iter(self, tbl1, tbl2):
        assert repr(list(tbl1)) == repr([tbl1.col1, tbl1.col2])
        assert repr(list(tbl2)) == repr([tbl2.col1, tbl2.col2, tbl2.col3])

        assert repr(list(tbl2 >> select(tbl2.col2))) == repr([tbl2.col2])

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

    def test_dict_constructor(self):
        t = pdt.Table({"a": [3, 1, 4, 1, 5, 9, 4]})
        assert_series_equal(
            (t >> mutate(r=pdt.rank(arrange=t.a))).r.export(Polars()),
            pl.Series([3, 1, 4, 1, 6, 7, 4]),
            check_names=False,
        )


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

        with pytest.raises(TypeError):
            assert 1 >> add(2) == 3

        add_10 = add(5) >> add(5)
        assert isinstance(add_10, Pipeable)

        with pytest.raises(TypeError):
            assert 5 >> add_10 == 15
