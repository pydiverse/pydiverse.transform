from __future__ import annotations

from pydiverse.transform._internal.ops.core import ElementWise, OperatorExtension, Unary
from pydiverse.transform._internal.ops.numeric import Add, Sub

__all__ = [
    "DtYear",
    "DtMonth",
    "DtDay",
    "DtHour",
    "DtMinute",
    "DtSecond",
    "DtMillisecond",
    "DtDayOfWeek",
    "DtDayOfYear",
    "DtDays",
    "DtHours",
    "DtMinutes",
    "DtSeconds",
    "DtMilliseconds",
    "DtSub",
    "DtDurAdd",
]


class DtExtract(ElementWise, Unary):
    signatures = ["datetime -> int64"]


class DateExtract(ElementWise, Unary):
    signatures = ["datetime -> int64", "date -> int64"]


class DtYear(DateExtract):
    name = "dt.year"


class DtMonth(DateExtract):
    name = "dt.month"


class DtDay(DateExtract):
    name = "dt.day"


class DtHour(DtExtract):
    name = "dt.hour"


class DtMinute(DtExtract):
    name = "dt.minute"


class DtSecond(DtExtract):
    name = "dt.second"


class DtMillisecond(DtExtract):
    name = "dt.millisecond"


class DtDayOfWeek(DateExtract):
    name = "dt.day_of_week"


class DtDayOfYear(DateExtract):
    name = "dt.day_of_year"


class DurationToUnit(ElementWise, Unary):
    signatures = ["duration -> int64"]


class DtDays(DurationToUnit):
    name = "dt.days"


class DtHours(DurationToUnit):
    name = "dt.hours"


class DtMinutes(DurationToUnit):
    name = "dt.minutes"


class DtSeconds(DurationToUnit):
    name = "dt.seconds"


class DtMilliseconds(DurationToUnit):
    name = "dt.milliseconds"


class DtSub(OperatorExtension):
    operator = Sub
    signatures = [
        "datetime, datetime -> duration",
        "datetime, date -> duration",
        "date, datetime -> duration",
        "date, date -> duration",
    ]


class DtDurAdd(OperatorExtension):
    operator = Add
    signatures = ["duration, duration -> duration"]
