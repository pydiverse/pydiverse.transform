from __future__ import annotations

import pydiverse.transform as pdt
from pydiverse.transform import C
from pydiverse.transform._internal.pipe.verbs import mutate
from pydiverse.transform._internal.tree.col_expr import LiteralCol
from tests.fixtures.backend import skip_backends
from tests.util import assert_result_equal


def test_count(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(**{col.name + "_count": col.count() for col in t})
        >> mutate(o=LiteralCol(0).count(filter=t.col3 == 2))
        >> mutate(u=pdt.count(), v=pdt.count(filter=t.col4 > 0)),
    )


def test_row_number(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            row_number=pdt.row_number(
                arrange=[C.col1.descending().nulls_first(), C.col5.nulls_last()]
            )
        ),
    )


# TODO: add these back in when string comparison is fixed in mssql
@skip_backends("mssql")
def test_min(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            # int1=pdt.min(C.col1 + 2, C.col2, 9),
            # int2=pdt.min(C.col1 * C.col2, 0),
            # int3=pdt.min(C.col1 * C.col2, C.col2 * C.col3, 2 - C.col3),
            int4=pdt.min(C.col1),
            # float1=pdt.min(C.col1, 1.5),
            # float2=pdt.min(1, C.col1 + 1.5, C.col2, 2.2),
            # str1=pdt.min(C.col5, "c"),
            # str2=pdt.min(C.col5, "C"),
        ),
    )


@skip_backends("mssql")
def test_max(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            int1=pdt.max(C.col1 + 2, C.col2, 9),
            int2=pdt.max(C.col1 * C.col2, 0),
            int3=pdt.max(C.col1 * C.col2, C.col2 * C.col3, 2 - C.col3),
            int4=pdt.max(C.col1),
            float1=pdt.max(C.col1, 1.5),
            float2=pdt.max(1, C.col1 + 1.5, C.col2, 2.2),
            str1=pdt.max(C.col5, "c"),
            str2=pdt.max(C.col5, "C"),
        ),
    )


def test_any_all(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            g=pdt.any(t.col1 == 0, t.col2 == 0),
            h=pdt.all(t.col1 < t.col4, t.col4 > 0, t.col5.str.len() <= 1),
        ),
    )


def test_sum(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            w=pdt.min(*(c for c in t if c.dtype().is_int())),
            u=pdt.sum(*(c for c in t if c.dtype().is_int())),
            h=pdt.sum(t.col1 * t.col4, t.col5.str.len() * 2, t.col2 - t.col1),
        ),
    )


def test_coalesce(df4):
    assert_result_equal(
        df4, lambda t: t >> mutate(w=pdt.coalesce(*(c for c in t))), exception=TypeError
    )

    assert_result_equal(
        df4,
        lambda t: t >> mutate(w=pdt.coalesce(*(c for c in t if c.dtype().is_int()))),
    )
