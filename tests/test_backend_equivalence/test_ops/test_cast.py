from __future__ import annotations

import pydiverse.transform as pdt
from pydiverse.transform.pipe.c import C
from pydiverse.transform.pipe.verbs import mutate
from tests.test_backend_equivalence.test_ops.test_ops_numerical import add_nan_inf_cols
from tests.util.assertion import assert_result_equal


def test_string_to_float(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t >> mutate(u=t.c.cast(pdt.Float64())),
    )


def test_string_to_int(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t >> mutate(u=t.d.cast(pdt.Int())),
    )


def test_float_to_int(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{col.name: col.cast(pdt.Int()) for col in t}),
    )

    assert_result_equal(
        df_num,
        lambda t: t >> add_nan_inf_cols() >> mutate(u=C.inf.cast(pdt.Int())),
        exception=Exception,
        may_throw=True,
    )
    assert_result_equal(
        df_num,
        lambda t: t >> add_nan_inf_cols() >> mutate(u=C.nan.cast(pdt.Int())),
        exception=Exception,
        may_throw=True,
    )


def test_datetime_to_date(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t >> mutate(u=t.col1.cast(pdt.Date()), v=t.col2.cast(pdt.Date())),
    )


def test_int_to_string(df_int):
    assert_result_equal(
        df_int, lambda t: t >> mutate(**{c.name: c.cast(pdt.String()) for c in t})
    )


def test_float_to_string(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (lambda s: s >> mutate(**{c.name: c.cast(pdt.String()) for c in s}))
        >> (lambda s: s >> mutate(**{c.name: c.cast(pdt.Float64()) for c in s})),
    )
