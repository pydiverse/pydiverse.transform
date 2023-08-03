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
    name = "year"


class Month(DatetimeExtractComponent):
    name = "month"


class Day(DatetimeExtractComponent):
    name = "day"


class Hour(DatetimeExtractComponent):
    name = "hour"


class Minute(DatetimeExtractComponent):
    name = "minute"


class Second(DatetimeExtractComponent):
    name = "second"


class Millisecond(DatetimeExtractComponent):
    name = "millisecond"


class DayOfWeek(DatetimeExtractComponent):
    name = "day_of_week"


class DayOfYear(DatetimeExtractComponent):
    name = "day_of_year"
