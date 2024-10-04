from __future__ import annotations

from pydiverse.transform import C
from pydiverse.transform._internal.pipe.verbs import (
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
            x=C.col1.str.strip(),
            y=C.col2.str.strip(),
        ),
    )


def test_string_length(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.str.len(),
            y=C.col2.str.len(),
        ),
    )


def test_to_upper(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.str.to_upper(),
            y=C.col2.str.to_upper(),
        ),
    )


def test_to_lower(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.str.to_lower(),
            y=C.col2.str.to_lower(),
        ),
    )


def test_replace_all(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.str.replace_all(" ", "").str.replace_all("foo", "fOO"),
            y=C.col2.str.replace_all("Ab", "ab"),
        ),
    )


def test_starts_with(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.str.starts_with("foo") | C.col1.str.starts_with(" "),
            y=C.col2.str.starts_with("test") | C.col2.str.starts_with("Abra"),
            underscore=C.col1.str.starts_with("_"),
            percent=C.col2.str.starts_with("%"),
        ),
    )


def test_ends_with(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.str.ends_with("Bar") | C.col1.str.ends_with(" "),
            y=C.col2.str.ends_with("_%") | C.col2.str.ends_with("Bar"),
            percent=C.col1.str.ends_with("%"),
            underscore=C.col2.str.ends_with("_"),
        ),
    )


def test_contains(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.str.contains(" ") | C.col1.str.contains("Foo"),
            y=C.col2.str.contains("st_") | C.col2.str.contains("bar"),
            percent=C.col1.str.contains("%"),
            underscore=C.col2.str.contains("_"),
        ),
    )


def test_slice(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            u=t.col1.str.slice(1, 3),
            v=t.col2.str.slice(t.col1.str.len() % (t.col2.str.len() + 1), 42),
            w=t.col1.str.slice(2, t.col1.str.len()),
        ),
    )
