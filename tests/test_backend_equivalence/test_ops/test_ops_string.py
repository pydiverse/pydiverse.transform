from __future__ import annotations

from pydiverse.transform import λ
from pydiverse.transform.core.verbs import (
    filter,
    mutate,
)
from tests.test_backend_equivalence import assert_result_equal


def test_eq(df_strings):
    assert_result_equal(df_strings, lambda t: t >> filter(λ.col1 == " "))
    assert_result_equal(df_strings, lambda t: t >> filter(λ.col1 == "foo"))
    assert_result_equal(df_strings, lambda t: t >> filter(λ.col1 == λ.col2))


def test_strip(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=λ.col1.strip(),
            y=λ.col2.strip(),
        ),
    )


def test_string_length(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=λ.col1.len(),
            y=λ.col2.len(),
        ),
    )


def test_upper(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=λ.col1.upper(),
            y=λ.col2.upper(),
        ),
    )


def test_lower(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=λ.col1.lower(),
            y=λ.col2.lower(),
        ),
    )


def test_replace(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=λ.col1.replace(" ", "").replace("foo", "fOO"),
            y=λ.col2.replace("Ab", "ab"),
        ),
    )


def test_startswith(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=λ.col1.startswith("foo") | λ.col1.startswith(" "),
            y=λ.col2.startswith("test") | λ.col2.startswith("Abra"),
            underscore=λ.col1.startswith("_"),
            percent=λ.col2.startswith("%"),
        ),
    )


def test_endswith(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=λ.col1.endswith("Bar") | λ.col1.endswith(" "),
            y=λ.col2.endswith("_%") | λ.col2.endswith("Bar"),
            percent=λ.col1.endswith("%"),
            underscore=λ.col2.endswith("_"),
        ),
    )


def test_contains(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=λ.col1.contains(" ") | λ.col1.contains("Foo"),
            y=λ.col2.contains("st_") | λ.col2.contains("bar"),
            percent=λ.col1.contains("%"),
            underscore=λ.col2.contains("_"),
        ),
    )
