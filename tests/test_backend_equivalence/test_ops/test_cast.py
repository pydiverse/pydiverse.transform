# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import pydiverse.transform as pdt
from pydiverse.common import Float32, Float64, Int32, Int64
from pydiverse.transform._internal.pipe.c import C
from pydiverse.transform._internal.pipe.verbs import mutate
from tests.util.assertion import assert_result_equal


def test_string_to_float(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t >> mutate(u=t.c.cast(pdt.Float64())),
    )


def test_string_to_int(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t >> mutate(u=t.d.cast(pdt.Int64())),
    )


def test_float_to_int(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(
            **{
                col.name: pdt.when((col <= (1 << 63) - 1) & (col >= -(1 << 63)))
                .then(col)
                .otherwise(0)
                .cast(pdt.Float64())
                for col in t
            }
        ),
    )

    # all backends throw on out of range values
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{c.cast(pdt.Int64()) for c in t}),
        exception=Exception,
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
        >> mutate(**{c.name: c.cast(pdt.String()) for c in t})
        >> (lambda s: mutate(**{c.name: c.cast(pdt.Float64()) for c in s})),
    )


def test_datetime_to_string(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=t.col1.cast(pdt.String()),
            y=t.col2.cast(pdt.String()),
        )
        >> mutate(
            x=C.x.str.to_datetime(),
            y=C.y.str.to_datetime(),
        ),
    )


def test_date_to_string(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=t.col1.cast(pdt.Date()).cast(pdt.String()),
            y=t.col2.cast(pdt.Date()).cast(pdt.String()),
            z=t.cdate.cast(pdt.String()),
        ),
    )


@pytest.mark.xfail
def test_non_strict_cast(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            **{
                f"{col.name}_{dtype}": col.cast(dtype, strict=False)
                for col in t
                for dtype in (Int64(), Int32(), Float64(), Float32())
            }
        ),
    )
