from __future__ import annotations

import pydiverse.transform as pdt
from pydiverse.transform.pipe.verbs import mutate
from tests.util.assertion import assert_result_equal


def test_exp(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(**{c.name: pdt.max(-700, pdt.min(700, c)).exp() for c in t}),
    )


def test_log(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{c.name: pdt.max(1e-16, c).log() for c in t}),
    )


def test_abs(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{c.name: abs(c) for c in t}),
    )


def test_round(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{c.name: round(c) for c in t}),
    )


def test_add(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(**{f"add_{c.name}_{d.name}": c + d for d in t for c in t}),
    )


def test_sub(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(**{f"sub_{c.name}_{d.name}": c - d for d in t for c in t}),
    )


def test_neg(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{f"neg_{c.name}": -c for c in t}),
    )


def test_mul(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(**{f"mul_{c.name}_{d.name}": c * d for d in t for c in t}),
    )


def test_div(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(
            **{c.name: pdt.when(abs(c) < 1e-50).then(1e-50).otherwise(c) for c in t}
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
        lambda t: t >> mutate(**{c.name: c.floor() for c in t}),
    )


def test_ceil(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{c.name: c.ceil() for c in t}),
    )
