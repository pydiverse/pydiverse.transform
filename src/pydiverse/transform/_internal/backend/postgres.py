from __future__ import annotations

import sqlalchemy as sqa

from pydiverse.transform._internal.backend.sql import SqlImpl
from pydiverse.transform._internal.ops import ops
from pydiverse.transform._internal.tree.col_expr import Cast, ColFn
from pydiverse.transform._internal.tree.types import Float, Int, String


class PostgresImpl(SqlImpl):
    @classmethod
    def compile_cast(cls, cast: Cast, sqa_col: dict[str, sqa.Label]) -> Cast:
        compiled_val = cls.compile_col_expr(cast.val, sqa_col)

        if cast.val.dtype() <= Float():
            if cast.target_type <= Int():
                return sqa.func.trunc(compiled_val).cast(sqa.BigInteger())

            if cast.target_type == String():
                compiled = sqa.cast(compiled_val, sqa.String)
                return sqa.case(
                    (compiled == "NaN", "nan"),
                    (compiled == "Infinity", "inf"),
                    (compiled == "-Infinity", "-inf"),
                    else_=compiled,
                )

        return sqa.cast(compiled_val, cls.sqa_type(cast.target_type))

    @classmethod
    def past_over_clause(
        cls, fn: ColFn, val: sqa.ColumnElement, *args: sqa.ColumnElement
    ) -> sqa.ColumnElement:
        if isinstance(fn.op, ops.DatetimeExtract | ops.DateExtract):
            return sqa.cast(val, sqa.BigInteger)
        elif fn.op == ops.sum:
            # postgres sometimes switches types for `sum`
            return sqa.cast(val, args[0].type)
        return val


with PostgresImpl.impl_store.impl_manager as impl:

    @impl(ops.less_than, String(), String())
    def _lt(x, y):
        return x < sqa.collate(y, "POSIX")

    @impl(ops.less_equal, String(), String())
    def _le(x, y):
        return x <= sqa.collate(y, "POSIX")

    @impl(ops.greater_than, String(), String())
    def _gt(x, y):
        return x > sqa.collate(y, "POSIX")

    @impl(ops.greater_equal, String(), String())
    def _ge(x, y):
        return x >= sqa.collate(y, "POSIX")

    @impl(ops.round)
    def _round(x, decimals=0):
        if decimals == 0:
            if isinstance(x.type, sqa.Integer):
                return x
            return sqa.func.ROUND(x, type_=x.type)

        if isinstance(x.type, sqa.Float):
            # Postgres doesn't support rounding of doubles to specific precision
            # -> Must first cast to numeric
            return sqa.func.ROUND(sqa.cast(x, sqa.Numeric), decimals, type_=sqa.Numeric)

        return sqa.func.ROUND(x, decimals, type_=x.type)

    @impl(ops.dt_second)
    def _dt_second(x):
        return sqa.func.FLOOR(sqa.extract("second", x), type_=sqa.Integer())

    @impl(ops.dt_millisecond)
    def _dt_millisecond(x):
        _1000 = sqa.literal_column("1000")
        return sqa.func.FLOOR(
            sqa.extract("milliseconds", x) % _1000, type_=sqa.Integer()
        )

    @impl(ops.horizontal_max, String(), String(), ...)
    def _horizontal_max(*x):
        return sqa.func.GREATEST(*(sqa.collate(e, "POSIX") for e in x))

    @impl(ops.horizontal_min, String(), String(), ...)
    def _least(*x):
        return sqa.func.LEAST(*(sqa.collate(e, "POSIX") for e in x))

    @impl(ops.any)
    def _any(x):
        return sqa.func.BOOL_OR(x, type_=sqa.Boolean())

    @impl(ops.all)
    def _all(x):
        return sqa.func.BOOL_AND(x, type_=sqa.Boolean())

    @impl(ops.is_nan)
    def _is_nan(x):
        return x == PostgresImpl.nan()

    @impl(ops.is_not_nan)
    def _is_not_nan(x):
        return x != PostgresImpl.nan()
