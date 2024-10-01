from __future__ import annotations

from pydiverse.transform.pipe.verbs import mutate
from tests.util.assertion import assert_result_equal


def test_add(df_int):
    assert_result_equal(
        df_int,
        lambda t: t
        >> (
            lambda s: mutate(**{f"add_{c.name}_{d.name}": c + d for d in s for c in s})
        ),
    )


def test_sub(df_int):
    assert_result_equal(
        df_int,
        lambda t: t
        >> (
            lambda s: (
                mutate(**{f"sub_{c.name}_{d.name}": c - d for d in s for c in s})
                >> (lambda u: mutate())
            )
        ),
    )


def test_neg(df_int):
    assert_result_equal(
        df_int,
        lambda t: t >> (lambda s: mutate(**{f"neg_{c.name}": -c for c in s})),
    )


def test_mul(df_int):
    assert_result_equal(
        df_int,
        lambda t: t
        >> (
            lambda s: mutate(**{f"mul_{c.name}_{d.name}": c * d for d in s for c in s})
        ),
    )


def test_truediv(df_int):
    assert_result_equal(
        df_int,
        lambda t: t
        >> (
            lambda s: mutate(**{f"div_{c.name}_{d.name}": c / d for d in s for c in s})
        ),
    )


def test_floordiv(df_int):
    assert_result_equal(
        df_int,
        lambda t: t
        >> (
            lambda s: mutate(**{f"div_{c.name}_{d.name}": c // d for d in s for c in s})
        ),
    )
