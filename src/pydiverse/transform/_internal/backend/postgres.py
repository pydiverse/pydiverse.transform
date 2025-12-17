# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import sqlalchemy as sqa
from sqlalchemy.dialects.postgresql import aggregate_order_by

from pydiverse.common import Bool, Dtype, Float, Float32, Int, Int32, Int64, String
from pydiverse.transform._internal.backend.sql import SqlImpl
from pydiverse.transform._internal.ops import ops
from pydiverse.transform._internal.tree import types
from pydiverse.transform._internal.tree.col_expr import Cast, ColFn


class PostgresImpl(SqlImpl):
    backend_name = "postgres"

    @classmethod
    def compile_cast(cls, cast: Cast, sqa_col: dict[str, sqa.Label]) -> Cast:
        if types.without_const(cast.val.dtype()).is_float():
            if cast.target_type.is_int():
                return cls.cast_compiled(cast, sqa.func.trunc(cls.compile_col_expr(cast.val, sqa_col)))

            if cast.target_type == String():
                compiled = super().compile_cast(cast, sqa_col)
                return sqa.case(
                    (compiled == "NaN", "nan"),
                    (compiled == "Infinity", "inf"),
                    (compiled == "-Infinity", "-inf"),
                    else_=compiled,
                )

        if types.without_const(cast.val.dtype()) == Bool() and cast.target_type == Int64():
            # postgres does not like casts bool -> bigint, so we go via int
            return cls.compile_cast(Cast(cast.val, Int32()).cast(Int64()), sqa_col)

        return super().compile_cast(cast, sqa_col)

    @classmethod
    def cast_compiled(cls, cast: Cast, compiled_expr: sqa.ColumnElement) -> sqa.Cast:
        if not cast.strict:
            cast.strict = True  # postgres does not have TRY_CAST

            if types.without_const(cast.val.dtype()) == String():
                compiled_expr = sqa.case(
                    (
                        sqa.func.pg_input_is_valid(
                            compiled_expr,
                            str(cls.sqa_type(cast.target_type)).lower().replace(" ", ""),
                        ),
                        compiled_expr,
                    ),
                    else_=sqa.null(),
                )

            def int_type_range(dtype: Int) -> tuple[int, int]:
                is_signed = dtype.__class__.__name__[0] == "I"
                bits = int(dtype.__class__.__name__[4 - is_signed :])

                if is_signed:
                    return (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
                else:
                    return (0, 2**bits - 1)

            if cast.val.dtype().is_int() and cast.target_type.is_int():
                source_range = int_type_range(cast.val.dtype())
                target_range = int_type_range(cast.target_type)

                if source_range[0] < target_range[0]:
                    compiled_expr = sqa.case(
                        (compiled_expr >= target_range[0], compiled_expr),
                        else_=sqa.null(),
                    )
                if source_range[1] > target_range[1]:
                    compiled_expr = sqa.case(
                        (compiled_expr <= target_range[1], compiled_expr),
                        else_=sqa.null(),
                    )

            if cast.val.dtype().is_float() and cast.target_type.is_int():
                target_range = int_type_range(cast.target_type)

                compiled_expr = sqa.case(
                    (
                        (compiled_expr != cls.nan())
                        & (compiled_expr != cls.inf())
                        & (compiled_expr != -cls.inf())
                        & (compiled_expr >= target_range[0])
                        & (compiled_expr <= target_range[1]),
                        compiled_expr,
                    ),
                    else_=sqa.null(),
                )

        return super().cast_compiled(cast, compiled_expr)

    @classmethod
    def sqa_type(cls, pdt_type: Dtype):
        if isinstance(pdt_type, types.List):
            return sqa.types.ARRAY(item_type=cls.sqa_type())
        if isinstance(pdt_type, Float32):
            return sqa.REAL()

        return super().sqa_type(pdt_type)

    @classmethod
    def fix_fn_types(cls, fn: ColFn, val: sqa.ColumnElement, *args: sqa.ColumnElement) -> sqa.ColumnElement:
        if isinstance(fn.op, ops.DatetimeExtract | ops.DateExtract):
            return sqa.cast(val, sqa.BigInteger)
        elif fn.op in (ops.sum, ops.cum_sum):
            # postgres sometimes switches types for `sum`
            return sqa.cast(val, args[0].type)
        return val

    @classmethod
    def compile_ordered_aggregation(
        cls, *args: sqa.ColumnElement, order_by: list[sqa.UnaryExpression], impl
    ) -> sqa.ColumnElement:
        return impl(*args[:-1], aggregate_order_by(args[-1], *order_by))


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
            return sqa.func.ROUND(sqa.cast(x, sqa.Numeric), decimals, type_=sqa.Numeric).cast(x.type)

        return sqa.func.ROUND(x, decimals, type_=x.type)

    @impl(ops.dt_second)
    def _dt_second(x):
        return sqa.func.FLOOR(sqa.extract("second", x), type_=sqa.Integer())

    @impl(ops.dt_millisecond)
    def _dt_millisecond(x):
        return sqa.func.FLOOR(sqa.extract("millisecond", x), type_=sqa.Integer()) % 1000

    @impl(ops.dt_microsecond)
    def _dt_microsecond(x):
        return sqa.func.FLOOR(sqa.extract("microsecond", x), type_=sqa.Integer()) % 1_000_000

    @impl(ops.horizontal_max, String(), String(), ...)
    def _horizontal_max(*x):
        return sqa.func.GREATEST(*(sqa.collate(e, "POSIX") for e in x))

    @impl(ops.horizontal_min, String(), String(), ...)
    def _least(*x):
        return sqa.func.LEAST(*(sqa.collate(e, "POSIX") for e in x))

    @impl(ops.any)
    @impl(ops.max, Bool())
    def _any(x):
        return sqa.func.BOOL_OR(x, type_=sqa.Boolean())

    @impl(ops.all)
    @impl(ops.min, Bool())
    def _all(x):
        return sqa.func.BOOL_AND(x, type_=sqa.Boolean())

    @impl(ops.is_nan)
    def _is_nan(x):
        return x == PostgresImpl.nan()

    @impl(ops.is_not_nan)
    def _is_not_nan(x):
        return x != PostgresImpl.nan()

    @impl(ops.dur_days)
    def _dur_days(x):
        sqa.func.extract("DAYS", x)

    @impl(ops.list_agg)
    def _list_agg(x):
        return sqa.func.array_agg(x)

    @impl(ops.truediv, Int(), Int())
    @impl(ops.truediv, Float(), Float())
    def _truediv(x, y):
        if not isinstance(x.type, Float):
            x = sqa.cast(x, sqa.Double)
        if not isinstance(y.type, Float):
            y = sqa.cast(y, sqa.Double)
        return x / y

    @impl(ops.dt_day_of_week)
    def _day_of_week(x):
        return (sqa.extract("dow", x) + 6) % sqa.literal_column("7") + 1

    @impl(ops.str_contains)
    def _str_contains(x, pattern, allow_regex, true_if_regex_unsupported):
        if not allow_regex:
            return x.contains(pattern, autoescape=True)
        return x.op("~")(pattern)
