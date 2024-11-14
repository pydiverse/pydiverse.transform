from __future__ import annotations

import sqlalchemy as sqa

from pydiverse.transform._internal.backend.sql import SqlImpl
from pydiverse.transform._internal.errors import NotSupportedError
from pydiverse.transform._internal.ops import ops
from pydiverse.transform._internal.tree.col_expr import Cast, ColFn
from pydiverse.transform._internal.tree.types import (
    Date,
    Datetime,
    Decimal,
    Float,
    String,
)
from pydiverse.transform._internal.util.warnings import warn_non_standard


class SqliteImpl(SqlImpl):
    @classmethod
    def inf(cls):
        return sqa.type_coerce(sqa.literal("1e314"), sqa.Double)

    @classmethod
    def nan(cls):
        raise NotSupportedError("SQLite does not have `nan`, use `null` instead.")

    @classmethod
    def compile_cast(cls, cast: Cast, sqa_col: dict[str, sqa.Label]) -> sqa.Cast:
        compiled_val = cls.compile_col_expr(cast.val, sqa_col)

        if cast.val.dtype() <= String() and cast.target_type <= Float():
            return sqa.case(
                (compiled_val == "inf", cls.inf()),
                (compiled_val == "-inf", -cls.inf()),
                else_=sqa.cast(
                    compiled_val,
                    cls.sqa_type(cast.target_type),
                ),
            )

        elif cast.val.dtype() <= Datetime() and cast.target_type == Date():
            return sqa.type_coerce(sqa.func.date(compiled_val), sqa.Date())

        elif cast.val.dtype() <= Float() and cast.target_type == String():
            return sqa.case(
                (compiled_val == cls.inf(), "inf"),
                (compiled_val == -cls.inf(), "-inf"),
                else_=sqa.cast(compiled_val, sqa.String),
            )

        return sqa.cast(compiled_val, cls.sqa_type(cast.target_type))

    @classmethod
    def past_over_clause(
        cls, fn: ColFn, val: sqa.ColumnElement, *args: sqa.ColumnElement
    ) -> sqa.ColumnElement:
        if (
            fn.op
            in (ops.horizontal_min, ops.horizontal_max, ops.mean, ops.min, ops.max)
            and fn.dtype() <= Float()
        ):
            return sqa.cast(val, sqa.Double)
        return val


with SqliteImpl.impl_store.impl_manager as impl:

    @impl(ops.round, Decimal())
    def _round(x, decimals=0):
        if decimals >= 0:
            return sqa.func.ROUND(x, decimals, type_=x.type)
        # For some reason SQLite doesn't like negative decimals values
        return sqa.func.ROUND(x / (10**-decimals), type_=x.type) * (10**-decimals)

    @impl(ops.str_starts_with)
    def _str_starts_with(x, y):
        warn_non_standard(
            "SQLite: startswith is case-insensitive by default. "
            "Use the 'case_sensitive_like' pragma to change this behaviour. "
            "See https://www.sqlite.org/pragma.html#pragma_case_sensitive_like",
        )
        return x.startswith(y, autoescape=True)

    @impl(ops.str_ends_with)
    def _str_ends_with(x, y):
        warn_non_standard(
            "SQLite: endswith is case-insensitive by default. "
            "Use the 'case_sensitive_like' pragma to change this behaviour. "
            "See https://www.sqlite.org/pragma.html#pragma_case_sensitive_like",
        )
        return x.endswith(y, autoescape=True)

    @impl(ops.str_contains)
    def _str_contains(x, y):
        warn_non_standard(
            "SQLite: contains is case-insensitive by default. "
            "Use the 'case_sensitive_like' pragma to change this behaviour. "
            "See https://www.sqlite.org/pragma.html#pragma_case_sensitive_like",
        )
        return x.contains(y, autoescape=True)

    @impl(ops.dt_millisecond)
    def _dt_millisecond(x):
        warn_non_standard(
            "SQLite returns rounded milliseconds",
        )
        _1000 = sqa.literal_column("1000")
        frac_seconds = sqa.cast(sqa.func.STRFTIME("%f", x), sqa.Numeric())
        return sqa.cast((frac_seconds * _1000) % _1000, sqa.Integer())

    @impl(ops.horizontal_max)
    def _greatest(*x):
        # The SQLite MAX function returns NULL if any of the inputs are NULL
        # -> Use divide and conquer approach with coalesce to ensure correct result
        if len(x) == 1:
            return x[0]

        mid = (len(x) + 1) // 2
        left = _greatest(*x[:mid])
        right = _greatest(*x[mid:])

        return sqa.func.coalesce(sqa.func.MAX(left, right), left, right)

    @impl(ops.horizontal_min)
    def _least(*x):
        # The SQLite MIN function returns NULL if any of the inputs are NULL
        # -> Use divide and conquer approach with coalesce to ensure correct result
        if len(x) == 1:
            return x[0]

        mid = (len(x) + 1) // 2
        left = _least(*x[:mid])
        right = _least(*x[mid:])

        return sqa.func.coalesce(sqa.func.MIN(left, right), left, right)

    # TODO: we need to get the string in the right format here (so sqlite can work with
    # it)
    @impl(ops.str_to_datetime)
    def _str_to_datetime(x):
        return sqa.type_coerce(x, sqa.DateTime)

    # the SQLite floor function is cursed... it throws if you pass in a large value
    # like 1e19. surprisingly, 1e18 works... what a coincidence... :)
    @impl(ops.floor)
    def _floor(x):
        return -sqa.func.ceil(-x)

    @impl(ops.is_nan)
    def _is_nan(x):
        return False

    @impl(ops.is_not_nan)
    def _is_not_nan(x):
        return True
