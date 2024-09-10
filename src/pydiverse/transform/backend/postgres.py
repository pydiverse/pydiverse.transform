from __future__ import annotations

import sqlalchemy as sa

from pydiverse.transform import ops
from pydiverse.transform.backend.sql import SqlImpl


class PostgresImpl(SqlImpl):
    dialect_name = "postgresql"


with PostgresImpl.op(ops.Less()) as op:

    @op("str, str -> bool")
    def _lt(x, y):
        return x < y.collate("POSIX")


with PostgresImpl.op(ops.LessEqual()) as op:

    @op("str, str -> bool")
    def _le(x, y):
        return x <= y.collate("POSIX")


with PostgresImpl.op(ops.Greater()) as op:

    @op("str, str -> bool")
    def _gt(x, y):
        return x > y.collate("POSIX")


with PostgresImpl.op(ops.GreaterEqual()) as op:

    @op("str, str -> bool")
    def _ge(x, y):
        return x >= y.collate("POSIX")


with PostgresImpl.op(ops.Round()) as op:

    @op.auto
    def _round(x, decimals=0):
        if decimals == 0:
            if isinstance(x.type, sa.Integer):
                return x
            return sa.func.ROUND(x, type_=x.type)

        if isinstance(x.type, sa.Float):
            # Postgres doesn't support rounding of doubles to specific precision
            # -> Must first cast to numeric
            return sa.func.ROUND(sa.cast(x, sa.Numeric), decimals, type_=sa.Numeric)

        return sa.func.ROUND(x, decimals, type_=x.type)


with PostgresImpl.op(ops.DtSecond()) as op:

    @op.auto
    def _second(x):
        return sa.func.FLOOR(sa.extract("second", x), type_=sa.Integer())


with PostgresImpl.op(ops.DtMillisecond()) as op:

    @op.auto
    def _millisecond(x):
        _1000 = sa.literal_column("1000")
        return sa.func.FLOOR(sa.extract("milliseconds", x) % _1000, type_=sa.Integer())


with PostgresImpl.op(ops.Greatest()) as op:

    @op("str... -> str")
    def _greatest(*x):
        # TODO: Determine return type
        return sa.func.GREATEST(*(e.collate("POSIX") for e in x))


with PostgresImpl.op(ops.Least()) as op:

    @op("str... -> str")
    def _least(*x):
        # TODO: Determine return type
        return sa.func.LEAST(*(e.collate("POSIX") for e in x))


with PostgresImpl.op(ops.Any()) as op:

    @op.auto
    def _any(x, *, _window_partition_by=None, _window_order_by=None):
        return sa.func.coalesce(sa.func.BOOL_OR(x, type_=sa.Boolean()), sa.false())

    @op.auto(variant="window")
    def _any(x, *, partition_by=None, order_by=None):
        return sa.func.coalesce(
            sa.func.BOOL_OR(x, type_=sa.Boolean()).over(
                partition_by=partition_by,
                order_by=order_by,
            ),
            sa.false(),
        )


with PostgresImpl.op(ops.All()) as op:

    @op.auto
    def _all(x):
        return sa.func.coalesce(sa.func.BOOL_AND(x, type_=sa.Boolean()), sa.false())

    @op.auto(variant="window")
    def _all(x, *, partition_by=None, order_by=None):
        return sa.func.coalesce(
            sa.func.BOOL_AND(x, type_=sa.Boolean()).over(
                partition_by=partition_by,
                order_by=order_by,
            ),
            sa.false(),
        )
