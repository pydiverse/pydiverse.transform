from __future__ import annotations

from pydiverse.transform import C
from pydiverse.transform.core.verbs import (
    filter,
    mutate,
)
from tests.util import assert_result_equal


def test_eq(df_strings):
    assert_result_equal(df_strings, lambda t: t >> filter(C.col1 == " "))
    assert_result_equal(df_strings, lambda t: t >> filter(C.col1 == "foo"))
    assert_result_equal(df_strings, lambda t: t >> filter(C.col1 == C.col2))


def test_nq(df_strings):
    assert_result_equal(df_strings, lambda t: t >> filter(C.col1 != " "))
    assert_result_equal(df_strings, lambda t: t >> filter(C.col1 != "foo"))
    assert_result_equal(df_strings, lambda t: t >> filter(C.col1 != C.col2))


def test_lt(df_strings):
    assert_result_equal(df_strings, lambda t: t >> filter(C.col1 < " x"))
    assert_result_equal(df_strings, lambda t: t >> filter(C.col1 < "E"))
    assert_result_equal(df_strings, lambda t: t >> filter(C.col1 < C.col2))


def test_gt(df_strings):
    assert_result_equal(df_strings, lambda t: t >> filter(C.col1 > " x"))
    assert_result_equal(df_strings, lambda t: t >> filter(C.col1 > "E"))
    assert_result_equal(df_strings, lambda t: t >> filter(C.col1 > C.col2))


def test_le(df_strings):
    assert_result_equal(df_strings, lambda t: t >> filter(C.col1 <= C.col2))


def test_ge(df_strings):
    assert_result_equal(df_strings, lambda t: t >> filter(C.col1 >= C.col2))


def test_strip(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.strip(),
            y=C.col2.strip(),
        ),
    )


def test_string_length(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.len(),
            y=C.col2.len(),
        ),
    )


def test_upper(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.upper(),
            y=C.col2.upper(),
        ),
    )


def test_lower(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.lower(),
            y=C.col2.lower(),
        ),
    )


def test_replace(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.replace(" ", "").replace("foo", "fOO"),
            y=C.col2.replace("Ab", "ab"),
        ),
    )


def test_startswith(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.startswith("foo") | C.col1.startswith(" "),
            y=C.col2.startswith("test") | C.col2.startswith("Abra"),
            underscore=C.col1.startswith("_"),
            percent=C.col2.startswith("%"),
        ),
    )


def test_endswith(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.endswith("Bar") | C.col1.endswith(" "),
            y=C.col2.endswith("_%") | C.col2.endswith("Bar"),
            percent=C.col1.endswith("%"),
            underscore=C.col2.endswith("_"),
        ),
    )


def test_contains(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.contains(" ") | C.col1.contains("Foo"),
            y=C.col2.contains("st_") | C.col2.contains("bar"),
            percent=C.col1.contains("%"),
            underscore=C.col2.contains("_"),
        ),
    )
