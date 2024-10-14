from __future__ import annotations

from pydiverse.transform._internal.ops.classes.numeric import Add, Sub
from pydiverse.transform._internal.ops.operator import (
    ElementWise,
    Unary,
)
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.dtypes import Date, Datetime, Duration, Int64

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
]


class DtExtract(ElementWise, Unary):
    signatures = [Signature(Datetime, returns=Int64)]


class DateExtract(ElementWise, Unary):
    signatures = [
        Signature(Datetime, returns=Int64),
        Signature(Date, returns=Int64),
    ]


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
    signatures = [Signature(Duration, returns=Int64)]


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


Sub.signatures.extend(
    [
        Signature(Datetime, Datetime, returns=Duration),
        Signature(Datetime, Date, returns=Duration),
        Signature(Date, Datetime, returns=Duration),
        Signature(Date, Date, returns=Duration),
    ]
)

Add.signatures.append(Signature(Duration, Duration, returns=Duration))
