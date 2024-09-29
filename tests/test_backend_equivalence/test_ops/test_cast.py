from __future__ import annotations

import pydiverse.transform as pdt
from pydiverse.transform.pipe.verbs import mutate
from tests.util.assertion import assert_result_equal


def test_string_to_float(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t >> mutate(u=t.c.cast(pdt.Float64())),
    )


def test_string_to_int(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t >> mutate(u=t.d.cast(pdt.Int())),
    )


def test_float_to_int(df_num):
    assert_result_equal(
        df_num,
        lambda t: t >> mutate(**{c.name: c.cast(pdt.Int()) for c in t}),
    )
