from __future__ import annotations

from pydiverse.transform._internal.pipe.c import C
from pydiverse.transform._internal.pipe.verbs import mutate, summarize
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
        >> mutate(**{c.name: c.map({0: 1}, default=c) for c in t})
        >> (
            lambda s: mutate(**{f"div_{c.name}_{d.name}": c / d for d in s for c in s})
        ),
    )


def test_floordiv(df_int):
    assert_result_equal(
        df_int,
        lambda t: t
        >> mutate(**{c.name: c.map({0: 1}, default=c) for c in t})
        >> (
            lambda s: mutate(**{f"div_{c.name}_{d.name}": c // d for d in s for c in s})
        ),
    )


def test_mod(df_int):
    assert_result_equal(
        df_int,
        lambda t: t
        >> mutate(**{c.name: c.map({0: 1}, default=c) for c in t})
        >> (
            lambda s: mutate(**{f"mod_{c.name}_{d.name}": c % d for d in s for c in s})
        ),
    )

    assert_result_equal(
        df_int,
        lambda t: t
        >> mutate(**{c.name: c.map({0: 1}, default=c) for c in t})
        >> (
            lambda s: mutate(**{f"mod_{c.name}_{d.name}": c % d for d in s for c in s})
            >> mutate(**{f"div_{c.name}_{d.name}": c // d for d in s for c in s})
        )
        >> summarize(
            **{
                f"div_plus_mod_{c.name}_{d.name}": (
                    (
                        C[f"div_{c.name}_{d.name}"] * C[d.name]
                        + C[f"mod_{c.name}_{d.name}"]
                    )
                    == C[c.name]
                ).all()
                for d in t
                for c in t
            }
        ),
    )
