from __future__ import annotations

from datetime import datetime

from pydiverse.transform import C
from pydiverse.transform.core.verbs import (
    filter,
    mutate,
)
from tests.util import assert_result_equal


def test_eq(df_datetime):
    assert_result_equal(
        df_datetime, lambda t: t >> filter(C.col1 == datetime(1970, 1, 1))
    )
    assert_result_equal(
        df_datetime, lambda t: t >> filter(C.col1 == datetime(2004, 12, 31))
    )
    assert_result_equal(df_datetime, lambda t: t >> filter(C.col1 == C.col2))


def test_nq(df_datetime):
    assert_result_equal(
        df_datetime, lambda t: t >> filter(C.col1 != datetime(1970, 1, 1))
    )
    assert_result_equal(
        df_datetime, lambda t: t >> filter(C.col1 != datetime(2004, 12, 31))
    )
    assert_result_equal(df_datetime, lambda t: t >> filter(C.col1 != C.col2))


def test_lt(df_datetime):
    assert_result_equal(
        df_datetime, lambda t: t >> filter(C.col1 < datetime(1970, 1, 1))
    )
    assert_result_equal(
        df_datetime, lambda t: t >> filter(C.col1 < datetime(2004, 12, 31))
    )
    assert_result_equal(df_datetime, lambda t: t >> filter(C.col1 < C.col2))


def test_gt(df_datetime):
    assert_result_equal(
        df_datetime, lambda t: t >> filter(C.col1 > datetime(1970, 1, 1))
    )
    assert_result_equal(
        df_datetime, lambda t: t >> filter(C.col1 > datetime(2004, 12, 31))
    )
    assert_result_equal(df_datetime, lambda t: t >> filter(C.col1 > C.col2))


def test_le(df_datetime):
    assert_result_equal(df_datetime, lambda t: t >> filter(C.col1 <= C.col2))


def test_ge(df_datetime):
    assert_result_equal(df_datetime, lambda t: t >> filter(C.col1 >= C.col2))


def test_year(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.year(),
            y=C.col2.year(),
        ),
    )


def test_month(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.month(),
            y=C.col2.month(),
        ),
    )


def test_day(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.day(),
            y=C.col2.day(),
        ),
    )


def test_hour(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.hour(),
            y=C.col2.hour(),
        ),
    )


def test_minute(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.minute(),
            y=C.col2.minute(),
        ),
    )


def test_second(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.second(),
            y=C.col2.second(),
        ),
    )


def test_millisecond(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.millisecond(),
            y=C.col2.millisecond(),
        ),
    )


def test_day_of_week(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.day_of_week(),
            y=C.col2.day_of_week(),
        ),
    )


def test_day_of_year(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.day_of_year(),
            y=C.col2.day_of_year(),
        ),
    )
