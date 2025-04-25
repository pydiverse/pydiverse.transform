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


def test_any_all(df_bool):
    assert_result_equal(df_bool, lambda t: t >> group_by(t.b) >> summarize(y=t.a.any()))
    assert_result_equal(df_bool, lambda t: t >> group_by(t.b) >> summarize(y=t.a.all()))


def test_bool_comparison(df_bool):
    assert_result_equal(df_bool, lambda t: t >> mutate(y=t.a < t.b))
    assert_result_equal(df_bool, lambda t: t >> mutate(y=t.a > t.b))
    assert_result_equal(df_bool, lambda t: t >> mutate(y=t.a <= t.b))
    assert_result_equal(df_bool, lambda t: t >> mutate(y=t.a >= t.b))


def test_bool_min_max(df_bool):
    assert_result_equal(df_bool, lambda t: t >> group_by(t.b) >> summarize(y=t.a.min()))
    assert_result_equal(df_bool, lambda t: t >> group_by(t.b) >> summarize(y=t.a.max()))


def test_bool_sum(df_bool):
    assert_result_equal(df_bool, lambda t: t >> mutate(y=t.a + t.b))
    assert_result_equal(df_bool, lambda t: t >> group_by(t.b) >> summarize(y=t.a.sum()))
    assert_result_equal(df_bool, lambda t: t >> group_by(t.a) >> summarize(y=t.b.sum()))
