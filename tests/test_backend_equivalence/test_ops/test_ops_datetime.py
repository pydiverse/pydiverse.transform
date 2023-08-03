from __future__ import annotations

from datetime import datetime

from pydiverse.transform import λ
from pydiverse.transform.core.verbs import (
    filter,
    mutate,
)
from tests.test_backend_equivalence import assert_result_equal


def test_eq(df_datetime):
    assert_result_equal(
        df_datetime, lambda t: t >> filter(λ.col1 == datetime(1970, 1, 1))
    )
    assert_result_equal(
        df_datetime, lambda t: t >> filter(λ.col1 == datetime(2004, 12, 31))
    )
    assert_result_equal(df_datetime, lambda t: t >> filter(λ.col1 == λ.col2))


def test_nq(df_datetime):
    assert_result_equal(
        df_datetime, lambda t: t >> filter(λ.col1 != datetime(1970, 1, 1))
    )
    assert_result_equal(
        df_datetime, lambda t: t >> filter(λ.col1 != datetime(2004, 12, 31))
    )
    assert_result_equal(df_datetime, lambda t: t >> filter(λ.col1 != λ.col2))


def test_lt(df_datetime):
    assert_result_equal(
        df_datetime, lambda t: t >> filter(λ.col1 < datetime(1970, 1, 1))
    )
    assert_result_equal(
        df_datetime, lambda t: t >> filter(λ.col1 < datetime(2004, 12, 31))
    )
    assert_result_equal(df_datetime, lambda t: t >> filter(λ.col1 < λ.col2))


def test_gt(df_datetime):
    assert_result_equal(
        df_datetime, lambda t: t >> filter(λ.col1 > datetime(1970, 1, 1))
    )
    assert_result_equal(
        df_datetime, lambda t: t >> filter(λ.col1 > datetime(2004, 12, 31))
    )
    assert_result_equal(df_datetime, lambda t: t >> filter(λ.col1 > λ.col2))


def test_le(df_datetime):
    assert_result_equal(df_datetime, lambda t: t >> filter(λ.col1 <= λ.col2))


def test_ge(df_datetime):
    assert_result_equal(df_datetime, lambda t: t >> filter(λ.col1 >= λ.col2))


def test_year(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=λ.col1.year(),
            y=λ.col2.year(),
        ),
    )


def test_month(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=λ.col1.month(),
            y=λ.col2.month(),
        ),
    )


def test_day(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=λ.col1.day(),
            y=λ.col2.day(),
        ),
    )


def test_hour(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=λ.col1.hour(),
            y=λ.col2.hour(),
        ),
    )


def test_minute(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=λ.col1.minute(),
            y=λ.col2.minute(),
        ),
    )


def test_second(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=λ.col1.second(),
            y=λ.col2.second(),
        ),
    )


def test_millisecond(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=λ.col1.millisecond(),
            y=λ.col2.millisecond(),
        ),
    )


def test_day_of_week(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=λ.col1.day_of_week(),
            y=λ.col2.day_of_week(),
        ),
    )


def test_day_of_year(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=λ.col1.day_of_year(),
            y=λ.col2.day_of_year(),
        ),
    )
