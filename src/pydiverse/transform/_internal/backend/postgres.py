from __future__ import annotations

import sqlalchemy as sqa

from pydiverse.transform._internal import ops
from pydiverse.transform._internal.backend.sql import SqlImpl
from pydiverse.transform._internal.tree import dtypes
from pydiverse.transform._internal.tree.col_expr import Cast


class PostgresImpl(SqlImpl):
    @classmethod
    def compile_cast(cls, cast: Cast, sqa_col: dict[str, sqa.Label]) -> Cast:
        compiled_val = cls.compile_col_expr(cast.val, sqa_col)

        if isinstance(cast.val.dtype(), dtypes.Float64):
            if isinstance(cast.target_type, dtypes.Int64):
                return sqa.func.trunc(compiled_val).cast(sqa.BigInteger())

            if isinstance(cast.target_type, dtypes.String):
                compiled = sqa.cast(compiled_val, sqa.String)
                return sqa.case(
                    (compiled == "NaN", "nan"),
                    (compiled == "Infinity", "inf"),
                    (compiled == "-Infinity", "-inf"),
                    else_=compiled,
                )

        return sqa.cast(compiled_val, cls.sqa_type(cast.target_type))


with PostgresImpl.op(ops.Less()) as op:

    @op("str, str -> bool")
    def _lt(x, y):
        return x < sqa.collate(y, "POSIX")


with PostgresImpl.op(ops.LessEqual()) as op:

    @op("str, str -> bool")
    def _le(x, y):
        return x <= sqa.collate(y, "POSIX")


with PostgresImpl.op(ops.Greater()) as op:

    @op("str, str -> bool")
    def _gt(x, y):
        return x > sqa.collate(y, "POSIX")


with PostgresImpl.op(ops.GreaterEqual()) as op:

    @op("str, str -> bool")
    def _ge(x, y):
        return x >= sqa.collate(y, "POSIX")


with PostgresImpl.op(ops.Round()) as op:

    @op.auto
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


with PostgresImpl.op(ops.DtSecond()) as op:

    @op.auto
    def _second(x):
        return sqa.func.FLOOR(sqa.extract("second", x), type_=sqa.Integer())


with PostgresImpl.op(ops.DtMillisecond()) as op:

    @op.auto
    def _millisecond(x):
        _1000 = sqa.literal_column("1000")
        return sqa.func.FLOOR(
            sqa.extract("milliseconds", x) % _1000, type_=sqa.Integer()
        )


with PostgresImpl.op(ops.Greatest()) as op:

    @op("str... -> str")
    def _greatest(*x):
        # TODO: Determine return type
        return sqa.func.GREATEST(*(sqa.collate(e, "POSIX") for e in x))


with PostgresImpl.op(ops.Least()) as op:

    @op("str... -> str")
    def _least(*x):
        # TODO: Determine return type
        return sqa.func.LEAST(*(sqa.collate(e, "POSIX") for e in x))


with PostgresImpl.op(ops.Any()) as op:

    @op.auto
    def _any(x, *, _window_partition_by=None, _window_order_by=None):
        return sqa.func.coalesce(sqa.func.BOOL_OR(x, type_=sqa.Boolean()), sqa.null())

    @op.auto(variant="window")
    def _any(x, *, partition_by=None, order_by=None):
        return sqa.func.coalesce(
            sqa.func.BOOL_OR(x, type_=sqa.Boolean()).over(
                partition_by=partition_by,
                order_by=order_by,
            ),
            sqa.null(),
        )


with PostgresImpl.op(ops.All()) as op:

    @op.auto
    def _all(x):
        return sqa.func.coalesce(sqa.func.BOOL_AND(x, type_=sqa.Boolean()), sqa.null())

    @op.auto(variant="window")
    def _all(x, *, partition_by=None, order_by=None):
        return sqa.func.coalesce(
            sqa.func.BOOL_AND(x, type_=sqa.Boolean()).over(
                partition_by=partition_by,
                order_by=order_by,
            ),
            sqa.null(),
        )


with SqlImpl.op(ops.IsNan()) as op:

    @op.auto
    def _is_nan(x):
        return x == PostgresImpl.nan()


with SqlImpl.op(ops.IsNotNan()) as op:

    @op.auto
    def _is_not_nan(x):
        return x != PostgresImpl.nan()
