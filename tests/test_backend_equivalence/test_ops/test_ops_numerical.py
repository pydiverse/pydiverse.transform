from __future__ import annotations

import pydiverse.transform as pdt
from pydiverse.transform.pipe.pipeable import verb
from pydiverse.transform.pipe.verbs import mutate
from tests.fixtures.backend import skip_backends
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


@skip_backends("postgres")
def test_exp(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (lambda s: mutate(**{c.name: c.exp() for c in s})),
    )


def test_exp_normal_inputs(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(
            **{
                c.name: pdt.when(c > 700)
                .then(700)
                .when(c < -700)
                .then(-700)
                .otherwise(c)
                for c in t
            }
        )
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
        >> (lambda s: mutate(**{c.name: abs(c) for c in s})),
    )


def test_round(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (lambda s: mutate(**{c.name: round(c) for c in s})),
    )


@skip_backends("postgres")
def test_add(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (
            lambda s: mutate(**{f"add_{c.name}_{d.name}": c + d for d in s for c in s})
        ),
    )


def test_add_normal_inputs(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(**{c.name: c / 3 for c in t})
        >> add_nan_inf_cols()
        >> (
            lambda s: mutate(**{f"add_{c.name}_{d.name}": c + d for d in s for c in s})
        ),
    )


@skip_backends("postgres")
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


def test_sub_normal_inputs(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(**{c.name: c / 3 for c in t})
        >> add_nan_inf_cols()
        >> (
            lambda s: mutate(**{f"sub_{c.name}_{d.name}": c - d for d in s for c in s})
        ),
    )


def test_neg(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (lambda s: mutate(**{f"neg_{c.name}": -c for c in s})),
    )


@skip_backends("postgres")
def test_mul(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> add_nan_inf_cols()
        >> (
            lambda s: mutate(**{f"mul_{c.name}_{d.name}": c * d for d in s for c in s})
        ),
    )


def test_mul_normal_inputs(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(**{c.name: abs(c) ** 0.4 for c in t})
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
            lambda s: mutate(
                **{c.name: pdt.when(c < 1e-10).then(1.0).otherwise(c) for c in s}
            )
        )
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
