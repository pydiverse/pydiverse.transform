from __future__ import annotations

from pydiverse.transform.pipe.verbs import mutate
from tests.util.assertion import assert_result_equal


def test_exp(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(
            exp_a=t.a.exp(),
            exp_b=t.b.exp(),
            exp_c=t.c.exp(),
            exp_d=t.d.exp(),
        ),
    )


def test_log(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(
            log_a=t.a.log(),
            log_b=t.b.log(),
            log_c=t.c.log(),
            log_d=t.d.log(),
            log_e=t.e.exp(),
        ),
    )


def test_abs(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(
            abs_a=abs(t.a),
            abs_b=abs(t.b),
            abs_c=abs(t.c),
            abs_d=abs(t.d),
        ),
    )


def test_round(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(
            round_a=round(t.a),
            round_b=round(t.b),
            round_c=round(t.c),
            round_d=round(t.d),
        ),
    )


def test_div(df_num):
    assert_result_equal(df_num, lambda t: t >> mutate(u=t.a / 2, v=t.b / 3.1))


def test_decimal(df_num):
    assert_result_equal(df_num, lambda t: t >> mutate(u=t.f + t.g, z=t.f * t.g))


def test_floor(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(
            u=t.a.floor(),
            v=t.b.floor(),
            w=t.f.floor(),
            x=t.d.floor(),
            y=t.e.floor(),
            z=t.f.floor(),
            q=t.g.floor(),
        ),
    )


def test_ceil(df_num):
    assert_result_equal(
        df_num,
        lambda t: t
        >> mutate(
            u=t.a.ceil(),
            v=t.b.ceil(),
            w=t.f.ceil(),
            x=t.d.ceil(),
            y=t.e.ceil(),
            z=t.f.ceil(),
            q=t.g.ceil(),
        ),
    )
