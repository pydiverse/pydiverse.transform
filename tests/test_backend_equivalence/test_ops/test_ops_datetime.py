from __future__ import annotations

from datetime import datetime

from pydiverse.transform import C
from pydiverse.transform._internal.pipe.verbs import (
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
            x=C.col1.dt.year(),
            y=C.col2.dt.year(),
            z=t.cdate.dt.year(),
        ),
    )


def test_month(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.dt.month(),
            y=C.col2.dt.month(),
            z=t.cdate.dt.month(),
        ),
    )


def test_day(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t >> mutate(x=C.col1.dt.day(), y=C.col2.dt.day(), z=t.cdate.dt.day()),
    )


def test_hour(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.dt.hour(),
            y=C.col2.dt.hour(),
        ),
    )

    assert_result_equal(
        df_datetime,
        lambda t: t >> mutate(z=t.cdate.dt.hour()),
        exception=TypeError,
    )


def test_minute(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.dt.minute(),
            y=C.col2.dt.minute(),
        ),
    )


def test_second(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.dt.second(),
            y=C.col2.dt.second(),
        ),
    )


def test_millisecond(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.dt.millisecond(),
            y=C.col2.dt.millisecond(),
        ),
    )


def test_day_of_week(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.dt.day_of_week(),
            y=C.col2.dt.day_of_week(),
        ),
    )


def test_day_of_year(df_datetime):
    assert_result_equal(
        df_datetime,
        lambda t: t
        >> mutate(
            x=C.col1.dt.day_of_year(),
            y=C.col2.dt.day_of_year(),
        ),
    )


# def test_duration_add(df_datetime):
#     assert_result_equal(df_datetime, lambda t: t >> mutate(z=t.cdur + t.cdur))


# def test_dt_subtract(df_datetime):
#     assert_result_equal(df_datetime, lambda t: t >> mutate(z=t.col1 - t.col2))
