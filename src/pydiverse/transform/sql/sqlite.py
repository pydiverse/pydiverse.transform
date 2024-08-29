from __future__ import annotations

import sqlalchemy as sa

from pydiverse.transform import ops
from pydiverse.transform.sql.sql_table import SQLTableImpl
from pydiverse.transform.util.warnings import warn_non_standard


class SQLiteTableImpl(SQLTableImpl):
    _dialect_name = "sqlite"


with SQLiteTableImpl.op(ops.Round()) as op:

    @op.auto
    def _round(x, decimals=0):
        if decimals >= 0:
            return sa.func.ROUND(x, decimals, type_=x.type)
        # For some reason SQLite doesn't like negative decimals values
        return sa.func.ROUND(x / (10**-decimals), type_=x.type) * (10**-decimals)


with SQLiteTableImpl.op(ops.StrStartsWith()) as op:

    @op.auto
    def _startswith(x, y):
        warn_non_standard(
            "SQLite: startswith is case-insensitive by default. "
            "Use the 'case_sensitive_like' pragma to change this behaviour. "
            "See https://www.sqlite.org/pragma.html#pragma_case_sensitive_like",
        )
        return x.startswith(y, autoescape=True)


with SQLiteTableImpl.op(ops.StrEndsWith()) as op:

    @op.auto
    def _endswith(x, y):
        warn_non_standard(
            "SQLite: endswith is case-insensitive by default. "
            "Use the 'case_sensitive_like' pragma to change this behaviour. "
            "See https://www.sqlite.org/pragma.html#pragma_case_sensitive_like",
        )
        return x.endswith(y, autoescape=True)


with SQLiteTableImpl.op(ops.StrContains()) as op:

    @op.auto
    def _contains(x, y):
        warn_non_standard(
            "SQLite: contains is case-insensitive by default. "
            "Use the 'case_sensitive_like' pragma to change this behaviour. "
            "See https://www.sqlite.org/pragma.html#pragma_case_sensitive_like",
        )
        return x.contains(y, autoescape=True)


with SQLiteTableImpl.op(ops.DtMillisecond()) as op:

    @op.auto
    def _millisecond(x):
        warn_non_standard(
            "SQLite returns rounded milliseconds",
        )
        _1000 = sa.literal_column("1000")
        frac_seconds = sa.cast(sa.func.STRFTIME("%f", x), sa.Numeric())
        return sa.cast((frac_seconds * _1000) % _1000, sa.Integer())


with SQLiteTableImpl.op(ops.Greatest()) as op:

    @op.auto
    def _greatest(*x):
        # The SQLite MAX function returns NULL if any of the inputs are NULL
        # -> Use divide and conquer approach with coalesce to ensure correct result
        if len(x) == 1:
            return x[0]

        mid = (len(x) + 1) // 2
        left = _greatest(*x[:mid])
        right = _greatest(*x[mid:])

        # TODO: Determine return type
        return sa.func.coalesce(sa.func.MAX(left, right), left, right)


with SQLiteTableImpl.op(ops.Least()) as op:

    @op.auto
    def _least(*x):
        # The SQLite MIN function returns NULL if any of the inputs are NULL
        # -> Use divide and conquer approach with coalesce to ensure correct result
        if len(x) == 1:
            return x[0]

        mid = (len(x) + 1) // 2
        left = _least(*x[:mid])
        right = _least(*x[mid:])

        # TODO: Determine return type
        return sa.func.coalesce(sa.func.MIN(left, right), left, right)
