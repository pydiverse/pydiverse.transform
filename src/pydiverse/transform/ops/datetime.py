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
    name = "dt.year"


class Month(DatetimeExtractComponent):
    name = "dt.month"


class Day(DatetimeExtractComponent):
    name = "dt.day"


class Hour(DatetimeExtractComponent):
    name = "dt.hour"


class Minute(DatetimeExtractComponent):
    name = "dt.minute"


class Second(DatetimeExtractComponent):
    name = "dt.second"


class Millisecond(DatetimeExtractComponent):
    name = "dt.millisecond"


class DayOfWeek(DatetimeExtractComponent):
    name = "dt.day_of_week"


class DayOfYear(DatetimeExtractComponent):
    name = "dt.day_of_year"
