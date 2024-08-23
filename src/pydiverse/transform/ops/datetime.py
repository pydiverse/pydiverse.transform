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


class DateExtractComponent(ElementWise, Unary):
    signatures = ["date -> int"]


class Year(DatetimeExtractComponent, DateExtractComponent):
    name = "dt.year"


class Month(DatetimeExtractComponent, DateExtractComponent):
    name = "dt.month"


class Day(DatetimeExtractComponent, DateExtractComponent):
    name = "dt.day"


class Hour(DatetimeExtractComponent):
    name = "dt.hour"


class Minute(DatetimeExtractComponent):
    name = "dt.minute"


class Second(DatetimeExtractComponent):
    name = "dt.second"


class Millisecond(DatetimeExtractComponent):
    name = "dt.millisecond"


class DayOfWeek(DatetimeExtractComponent, DateExtractComponent):
    name = "dt.day_of_week"


class DayOfYear(DatetimeExtractComponent, DateExtractComponent):
    name = "dt.day_of_year"
