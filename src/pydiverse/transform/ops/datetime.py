from __future__ import annotations

from pydiverse.transform.ops.core import ElementWise, OperatorExtension, Unary
from pydiverse.transform.ops.numeric import Add, RAdd, RSub, Sub

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
    "DtRSub",
    "DtDurAdd",
    "DtDurRAdd",
]


class DtExtract(ElementWise, Unary):
    signatures = ["datetime -> int"]


class DateExtract(ElementWise, Unary):
    signatures = ["date -> int"]


class DtYear(DtExtract, DateExtract):
    name = "dt.year"


class DtMonth(DtExtract, DateExtract):
    name = "dt.month"


class DtDay(DtExtract, DateExtract):
    name = "dt.day"


class DtHour(DtExtract):
    name = "dt.hour"


class DtMinute(DtExtract):
    name = "dt.minute"


class DtSecond(DtExtract):
    name = "dt.second"


class DtMillisecond(DtExtract):
    name = "dt.millisecond"


class DtDayOfWeek(DtExtract, DateExtract):
    name = "dt.day_of_week"


class DtDayOfYear(DtExtract, DateExtract):
    name = "dt.day_of_year"


class DurationToUnit(ElementWise, Unary):
    signatures = ["duration -> int"]


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


class DtRSub(OperatorExtension):
    operator = RSub
    signatures = [
        "datetime, datetime -> duration",
        "datetime, date -> duration",
        "date, datetime -> duration",
        "date, date -> duration",
    ]


class DtDurAdd(OperatorExtension):
    operator = Add
    signatures = ["duration, duration -> duration"]


class DtDurRAdd(OperatorExtension):
    operator = RAdd
    signatures = ["duration, duration -> duration"]
