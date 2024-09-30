from __future__ import annotations

import pydiverse.transform as pdt
from pydiverse.transform.pipe.pipeable import verb
from pydiverse.transform.pipe.verbs import mutate
from tests.util.assertion import assert_result_equal


@verb
def add_nan_inf_cols(table: pdt.Table) -> pdt.Table:
    return table >> mutate(
        **{
            "nan": float("nan"),
            "negnan": float("-nan"),
            "inf": float("inf"),
            "neginf": float("-inf"),
        }
    )


def test_exp(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (lambda s: mutate(**{c.name: c.exp() for c in s})),
    )


def test_log(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (lambda s: mutate(**{c.name: c.log() for c in s})),
    )


def test_abs(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (lambda s: mutate(**{c.name: c.abs() for c in s})),
    )


def test_round(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (lambda s: mutate(**{c.name: round(c) for c in s})),
    )


def test_add(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (
            lambda s: mutate(**{f"add_{c.name}_{d.name}": c + d for d in s for c in s})
        ),
    )


def test_sub(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (
            lambda s: (
                mutate(**{f"sub_{c.name}_{d.name}": c - d for d in s for c in s})
                >> (lambda u: mutate())
            )
        ),
    )


def test_neg(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (lambda s: mutate(**{f"neg_{c.name}": -c for c in s})),
    )


def test_mul(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (
            lambda s: mutate(**{f"mul_{c.name}_{d.name}": c * d for d in s for c in s})
        ),
    )


def test_div(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (
            lambda s: mutate(**{f"div_{c.name}_{d.name}": c / d for d in s for c in s})
        ),
    )


def test_decimal(df_num):
    # TODO: test the decimal here
    assert_result_equal(df_num, lambda t: t >> mutate(u=t.f + t.g, z=t.f * t.g))


def test_floor(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (lambda s: mutate(**{c.name: c.floor() for c in s})),
    )


def test_ceil(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (lambda s: mutate(**{c.name: c.ceil() for c in s})),
    )
