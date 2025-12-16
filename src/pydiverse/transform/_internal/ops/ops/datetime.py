# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import Date, Datetime, Duration, Int


class DatetimeExtract(Operator):
    def __init__(self, name: str, doc: str | None = None):
        super().__init__(
            name,
            Signature(Datetime(), return_type=Int()),
            doc=doc if doc is not None else f"Extracts the {name[3:]} component.",
        )


class DateExtract(Operator):
    def __init__(self, name: str, doc: str | None = None):
        super().__init__(
            name,
            Signature(Date(), return_type=Int()),
            Signature(Datetime(), return_type=Int()),
            doc=doc if doc is not None else f"Extracts the {name[3:]} component.",
        )


dt_year = DateExtract("dt.year", doc="The year component of the date or datetime.")

dt_month = DateExtract("dt.month", doc="The month component of the date or datetime.")

dt_day = DateExtract("dt.day", doc="The day component of the date or datetime.")

dt_hour = DatetimeExtract("dt.hour", doc="The hour component of the datetime.")

dt_minute = DatetimeExtract("dt.minute", doc="The minute component of the datetime.")

dt_second = DatetimeExtract("dt.second", doc="The second component of the datetime.")

dt_millisecond = DatetimeExtract(
    "dt.millisecond",
    doc="""
The microsecond component of the datetime int-divided by 1000 (see polars).
""",
)

dt_microsecond = DatetimeExtract("dt.microsecond", doc="The microsecond component of the datetime.")

dt_day_of_week = DateExtract(
    "dt.day_of_week",
    doc="""
The number of the current weekday.

This is one-based, so Monday is 1 and Sunday is 7.
""",
)

dt_day_of_year = DateExtract(
    "dt.day_of_year",
    doc="""
The number of days since the beginning of the year.

This is one-based, so it returns 1 for the 1st of January.
""",
)


class DurationToUnit(Operator):
    def __init__(self, name: str, doc: str = ""):
        super().__init__(name, Signature(Duration(), return_type=Int()), doc=doc)


dur_days = DurationToUnit("dur.days")

dur_hours = DurationToUnit("dur.hours")

dur_minutes = DurationToUnit("dur.minutes")

dur_seconds = DurationToUnit("dur.seconds")

dur_milliseconds = DurationToUnit("dur.milliseconds")

dur_microseconds = DurationToUnit("dur.microseconds")
