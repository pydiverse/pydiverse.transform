from __future__ import annotations

from pydiverse.transform.extended import *
from tests.util.assertion import assert_result_equal


def test_or(df_bool):
    assert_result_equal(df_bool, lambda t: t >> mutate(y=t.a | t.b))


def test_and(df_bool):
    assert_result_equal(df_bool, lambda t: t >> mutate(y=t.a & t.b))


def test_xor(df_bool):
    assert_result_equal(df_bool, lambda t: t >> mutate(y=t.a ^ t.b))


def test_invert(df_bool):
    assert_result_equal(df_bool, lambda t: t >> mutate(y=~t.a))
