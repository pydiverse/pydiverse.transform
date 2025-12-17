# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import pydiverse.transform as pdt
from pydiverse.transform._internal.pipe.c import C
from pydiverse.transform._internal.pipe.verbs import mutate, select
from pydiverse.transform.common import *
from tests.fixtures.backend import skip_backends
from tests.util.assertion import assert_result_equal


def test_exp(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{c.name: c.clip(-700, 700).exp() for c in t}),
    )


def test_log(df_num):
    assert_result_equal(df_num, lambda t: t >> mutate(**{c.name: pdt.max(1e-16, c).log() for c in t}))


def test_abs(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{c.name: c.abs() for c in t}),
    )


def test_round(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{c.name: c.round() for c in t}),
    )
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{c.name: c.round(2) for c in t}),
    )
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{c.name: c.round(-2) for c in t}),
    )


def test_add(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{f"add_{c.name}_{d.name}": c + d for d in t for c in t}),
    )


def test_sub(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{f"sub_{c.name}_{d.name}": c - d for d in t for c in t}),
    )


def test_neg(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{f"neg_{c.name}": -c for c in t}),
    )


def test_mul(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{f"mul_{c.name}_{d.name}": c * d for d in t for c in t}),
    )


def test_div(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(**{c.name: pdt.when(c.abs() < 1e-50).then(1e-50).otherwise(c) for c in t})
        >> (lambda s: mutate(**{f"div_{c.name}_{d.name}": c / d for d in s for c in s})),
    )


def test_decimal(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(f=t.f.cast(pdt.Decimal), g=t.g.cast(pdt.Decimal)) >> mutate(u=C.f + C.g, z=C.f * C.g),
    )


def test_floor(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{c.name: c.floor() for c in t}),
    )


def test_ceil(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{c.name: c.ceil() for c in t}),
    )


# MSSQL is really bad with inf/nan handling. IBM DB2 supports it with DECFLOAT but
# sqlalchemy doesn't support that.
@skip_backends("mssql", "ibm_db2")
def test_inf_lit(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> select()
        >> mutate(
            inf=float("inf"),
            neg_inf=float("-inf"),
        )
        >> mutate(
            inf_str=C.inf.cast(pdt.String()),
            neg_inf_str=C.neg_inf.cast(pdt.String()),
        )
        >> mutate(
            inf_back=C.inf_str.cast(pdt.Float64()),
            neg_inf_back=C.neg_inf_str.cast(pdt.Float64()),
        ),
    )


# MSSQL is really bad with inf/nan handling. IBM DB2 supports it with DECFLOAT but
# sqlalchemy doesn't support that.
@skip_backends("mssql", "sqlite", "ibm_db2")
def test_nan_lit(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> select()
        >> mutate(nan=float("nan"))
        >> mutate(nan_str=C.nan.cast(pdt.String()))
        >> mutate(nan_back=C.nan_str.cast(pdt.Float64())),
    )


# MSSQL is really bad with inf/nan handling. IBM DB2 supports it with DECFLOAT but
# sqlalchemy doesn't support that.
@skip_backends("mssql", "ibm_db2")
def test_is_inf(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(inf=float("inf"))
        >> mutate(neg_inf=-C.inf)
        >> mutate(
            inf_is_inf=C.inf.is_inf(),
            neg_inf_is_inf=C.neg_inf.is_inf(),
            **{c.name + "is_inf": c.is_inf() for c in t},
            inf_is_not_inf=C.inf.is_not_inf(),
            neg_inf_is_not_inf=C.neg_inf.is_not_inf(),
            **{c.name + "is_not_inf": c.is_not_inf() for c in t},
        ),
    )


# MSSQL is really bad with inf/nan handling. IBM DB2 supports it with DECFLOAT but
# sqlalchemy doesn't support that.
@skip_backends("mssql", "sqlite", "ibm_db2")
def test_is_nan(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(nan=float("nan"))
        >> mutate(
            nan_is_nan=C.nan.is_nan(),
            **{c.name + "is_nan": c.is_nan() for c in t},
            nan_is_not_nan=C.nan.is_not_nan(),
            **{c.name + "is_not_nan": c.is_not_nan() for c in t},
        ),
    )


def test_int_pow(df_int):
    assert_result_equal(df_int, lambda t: t >> mutate(u=pdt.min(t.a, 10) ** pdt.min(t.b.abs(), 5)))


def test_sin(df_num):
    assert_result_equal(df_num, lambda t: t >> mutate(**{c.name: c.sin() for c in t}))


def test_cos(df_num):
    assert_result_equal(df_num, lambda t: t >> mutate(**{c.name: c.cos() for c in t}))


def test_tan(df_num):
    assert_result_equal(df_num, lambda t: t >> mutate(**{c.name: c.atan().tan() for c in t}))


def test_asin(df_num):
    assert_result_equal(df_num, lambda t: t >> mutate(**{c.name: c.sin().asin() for c in t}))


def test_acos(df_num):
    assert_result_equal(df_num, lambda t: t >> mutate(**{c.name: c.cos().acos() for c in t}))


def test_atan(df_num):
    assert_result_equal(df_num, lambda t: t >> mutate(**{c.name: c.atan() for c in t}))


def test_sqrt(df_num):
    assert_result_equal(df_num, lambda t: t >> mutate(**{c.name: c.abs().sqrt() for c in t}))


def test_cbrt(df_num):
    assert_result_equal(df_num, lambda t: t >> mutate(**{c.name: c.cbrt() for c in t}))


def test_log10(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{c.name: pdt.max(1e-16, c).log10() for c in t}),
    )


def test_clip(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(
            **{c.name + "1": c.clip(0.1, 10) for c in t},
            **{c.name + "2": c.clip(-500, 1024) for c in t},
        ),
    )


def test_rand(df_num):
    # just see that it runs
    def f(t):
        t >> mutate(r=pdt.rand()) >> export(Polars)
        return pdt.Table({"a": [1]})

    assert_result_equal(df_num, f)
