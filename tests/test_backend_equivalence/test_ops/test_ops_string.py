from __future__ import annotations

from pydiverse.transform.extended import *
from tests.fixtures.backend import skip_backends
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
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            col1_le_c=C.col1 <= C.c,
            col1_le_col2=t.col1 <= t.col2,
            d_le_c=t.d <= t.c,
        ),
    )
    assert_result_equal(df_strings, lambda t: t >> filter(C.col1 <= C.col2))


def test_ge(df_strings):
    assert_result_equal(
        df_strings, lambda t: t >> mutate(col1_ge_col2=C.col1 >= C.col2)
    )
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


def test_upper(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.str.upper(),
            y=C.col2.str.upper(),
        ),
    )


def test_lower(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.str.lower(),
            y=C.col2.str.lower(),
        ),
    )


def test_replace_all(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> mutate(
            x=C.col1.str.replace_all(" ", "").str.replace_all("foo", "fOO"),
            y=C.col2.str.replace_all("Ab", "ab"),
            z=C.e.str.replace_all("abba", "#"),
            q=C.e.str.replace_all("--", "="),
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


def test_str_join(df_strings):
    assert_result_equal(
        df_strings,
        lambda t: t
        >> group_by(t.e)
        >> summarize(
            con=t.c.str.join(", ", arrange=[t.d.nulls_first(), t.c.nulls_last()])
        ),
    )

    assert_result_equal(
        df_strings,
        lambda t: t >> group_by(t.gb) >> summarize(y=t.col1.str.join(arrange=t.col2)),
    )


@skip_backends("mssql")
def test_str_arrange(df_strings):
    def bind(col):
        assert_result_equal(
            df_strings,
            lambda t: t >> arrange(t[col].nulls_last(), t.c.nulls_last()),
            check_row_order=True,
        )

    for col in ["col1", "col2", "c", "d", "e", "gb"]:
        bind(col)
