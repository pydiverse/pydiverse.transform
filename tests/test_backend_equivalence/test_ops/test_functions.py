from __future__ import annotations

from pydiverse.transform import C
from pydiverse.transform import functions as f
from pydiverse.transform.core.verbs import mutate
from tests.fixtures.backend import skip_backends
from tests.util import assert_result_equal


def test_count(df4):
    assert_result_equal(
        df4, lambda t: t >> mutate(**{col._.name + "_count": f.count(col) for col in t})
    )


def test_row_number(df4):
    assert_result_equal(
        df4,
        lambda t: t >> mutate(row_number=f.row_number(arrange=[-C.col1, C.col5])),
    )


# MSSQL Added the LEAST function in version 2022.
# Our docker container doesn't yet support it.
@skip_backends("mssql")
def test_min(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            int1=f.min(C.col1 + 2, C.col2, 9),
            int2=f.min(C.col1 * C.col2, 0),
            int3=f.min(C.col1 * C.col2, C.col2 * C.col3, 2 - C.col3),
            int4=f.min(C.col1),
            float1=f.min(C.col1, 1.5),
            float2=f.min(1, C.col1 + 1.5, C.col2, 2.2),
            str1=f.min(C.col5, "c"),
            str2=f.min(C.col5, "C"),
        ),
    )


# MSSQL Added the GREATEST function in version 2022.
# Our docker container doesn't yet support it.
@skip_backends("mssql")
def test_max(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(
            int1=f.max(C.col1 + 2, C.col2, 9),
            int2=f.max(C.col1 * C.col2, 0),
            int3=f.max(C.col1 * C.col2, C.col2 * C.col3, 2 - C.col3),
            int4=f.max(C.col1),
            float1=f.max(C.col1, 1.5),
            float2=f.max(1, C.col1 + 1.5, C.col2, 2.2),
            str1=f.max(C.col5, "c"),
            str2=f.max(C.col5, "C"),
        ),
    )
