from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import Date, Datetime, Duration, Int


class DatetimeExtract(Operator):
    def __init__(self, name: str):
        super().__init__(name, Signature(Datetime(), return_type=Int()))


class DateExtract(Operator):
    def __init__(self, name: str):
        super().__init__(
            name,
            Signature(Date(), return_type=Int()),
            Signature(Datetime(), return_type=Int()),
        )


dt_year = DateExtract("dt.year")

dt_month = DateExtract("dt.month")

dt_day = DateExtract("dt.day")

dt_hour = DatetimeExtract("dt.hour")

dt_minute = DatetimeExtract("dt.minute")

dt_second = DatetimeExtract("dt.second")

dt_millisecond = DatetimeExtract("dt.millisecond")

dt_microsecond = DatetimeExtract("dt.microsecond")

dt_day_of_week = DateExtract("dt.day_of_week")

dt_day_of_year = DateExtract("dt.day_of_year")


class DurationToUnit(Operator):
    def __init__(self, name: str):
        super().__init__(name, Signature(Duration(), return_type=Int()))


dt_days = DurationToUnit("dt.days")

dt_hours = DurationToUnit("dt.hours")

dt_minutes = DurationToUnit("dt.minutes")

dt_seconds = DurationToUnit("dt.seconds")

dt_milliseconds = DurationToUnit("dt.milliseconds")
