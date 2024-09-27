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
