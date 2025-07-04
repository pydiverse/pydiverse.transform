# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pydiverse.transform.extended import *
from tests.util import assert_result_equal


def test_noop(df2):
    assert_result_equal(df2, lambda t: t >> filter())
    assert_result_equal(df2, lambda t: t >> filter(t.col1 == t.col1))


def test_simple_filter(df2):
    assert_result_equal(df2, lambda t: t >> filter(t.col1 == 2))
    assert_result_equal(df2, lambda t: t >> filter(t.col1 != 2))


def test_chained_filters(df2):
    assert_result_equal(df2, lambda t: t >> filter(1 < t.col1) >> filter(t.col1 < 5))

    assert_result_equal(df2, lambda t: t >> filter(1 < t.col1) >> filter(t.col3 < 0.25))


def test_filter_empty_result(df3):
    assert_result_equal(
        df3,
        lambda t: t >> filter(t.col1 == 0) >> filter(t.col2 == 2) >> filter(t.col4 < 2),
    )


def test_filter_after_mutate(df4):
    assert_result_equal(
        df4,
        lambda t: t >> mutate(x=t.col1 <= t.col2) >> filter(C.x),
    )


def test_filter_is_in(df4):
    assert_result_equal(
        df4,
        lambda t: t
        >> filter(
            C.col1.is_in(0, 2),
            C.col2.is_in(0, t.col1 * t.col2),
        )
        >> mutate(u=C.col3.is_in()),
    )

    assert_result_equal(
        df4,
        lambda t: t >> filter((-(t.col4 // 2 - 1)).is_in(1, 4, t.col1 + t.col2)),
    )

    assert_result_equal(df4, lambda t: t >> filter(t.col1.is_in(None)))

    assert_result_equal(
        df4,
        lambda t: t
        >> mutate(x=t.col1.is_in(0, 2))
        >> filter(
            t.col2.is_in(0, 2) & C.x,
        ),
    )
