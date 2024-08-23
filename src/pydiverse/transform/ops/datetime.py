from __future__ import annotations

from pydiverse.transform.ops.core import ElementWise, Unary

__all__ = [
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "Second",
    "Millisecond",
    "DayOfWeek",
    "DayOfYear",
]


class DatetimeExtractComponent(ElementWise, Unary):
    signatures = ["datetime -> int"]


class Year(DatetimeExtractComponent):
    name = "dt_year"


class Month(DatetimeExtractComponent):
    name = "dt_month"


class Day(DatetimeExtractComponent):
    name = "dt_day"


class Hour(DatetimeExtractComponent):
    name = "dt_hour"


class Minute(DatetimeExtractComponent):
    name = "dt_minute"


class Second(DatetimeExtractComponent):
    name = "dt_second"


class Millisecond(DatetimeExtractComponent):
    name = "dt_millisecond"


class DayOfWeek(DatetimeExtractComponent):
    name = "dt_day_of_week"


class DayOfYear(DatetimeExtractComponent):
    name = "dt_day_of_year"
