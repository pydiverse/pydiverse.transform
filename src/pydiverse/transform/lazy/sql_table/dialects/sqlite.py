from __future__ import annotations

import warnings

import sqlalchemy as sa

from pydiverse.transform import ops
from pydiverse.transform.errors import NonStandardBehaviourWarning
from pydiverse.transform.lazy.sql_table.sql_table import SQLTableImpl


class SQLiteTableImpl(SQLTableImpl):
    _dialect_name = "sqlite"


with SQLiteTableImpl.op(ops.Round()) as op:

    @op.auto
    def _round(x, decimals=0):
        if decimals >= 0:
            return sa.func.ROUND(x, decimals, type_=x.type)
        # For some reason SQLite doesn't like negative decimals values
        return sa.func.ROUND(x / (10**-decimals), type_=x.type) * (10**-decimals)


with SQLiteTableImpl.op(ops.StartsWith()) as op:

    @op.auto
    def _startswith(x, y):
        warnings.warn(
            "SQLite: startswith is case-insensitive by default. "
            "Use the 'case_sensitive_like' pragma to change this behaviour. "
            "See https://www.sqlite.org/pragma.html#pragma_case_sensitive_like",
            NonStandardBehaviourWarning,
        )
        return x.startswith(y, autoescape=True)


with SQLiteTableImpl.op(ops.EndsWith()) as op:

    @op.auto
    def _endswith(x, y):
        warnings.warn(
            "SQLite: endswith is case-insensitive by default. "
            "Use the 'case_sensitive_like' pragma to change this behaviour. "
            "See https://www.sqlite.org/pragma.html#pragma_case_sensitive_like",
            NonStandardBehaviourWarning,
        )
        return x.endswith(y, autoescape=True)


with SQLiteTableImpl.op(ops.Contains()) as op:

    @op.auto
    def _contains(x, y):
        warnings.warn(
            "SQLite: contains is case-insensitive by default. "
            "Use the 'case_sensitive_like' pragma to change this behaviour. "
            "See https://www.sqlite.org/pragma.html#pragma_case_sensitive_like",
            NonStandardBehaviourWarning,
        )
        return x.contains(y, autoescape=True)


with SQLiteTableImpl.op(ops.StringJoin()) as op:

    @op.auto
    def _join(x, sep: str):
        return sa.func.GROUP_CONCAT(x, sep, type_=x.type)
