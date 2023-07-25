from __future__ import annotations

import sqlalchemy as sa

from pydiverse.transform import ops
from pydiverse.transform.lazy.sql_table.sql_table import SQLTableImpl


class PostgresTableImpl(SQLTableImpl):
    _dialect_name = "postgresql"


with PostgresTableImpl.op(ops.Any()) as op:

    @op.auto
    def _any(x, *, _window_partition_by=None, _window_order_by=None):
        return sa.func.coalesce(sa.func.BOOL_OR(x, type_=sa.Boolean()), sa.false())

    @op.auto(variant="window")
    def _any(x, *, _window_partition_by=None, _window_order_by=None):
        return sa.func.coalesce(
            sa.func.BOOL_OR(x, type_=sa.Boolean()).over(
                partition_by=_window_partition_by,
                order_by=_window_order_by,
            ),
            sa.false(),
        )


with PostgresTableImpl.op(ops.All()) as op:

    @op.auto
    def _all(x):
        return sa.func.coalesce(sa.func.BOOL_AND(x, type_=sa.Boolean()), sa.false())

    @op.auto(variant="window")
    def _all(x, *, _window_partition_by=None, _window_order_by=None):
        return sa.func.coalesce(
            sa.func.BOOL_AND(x, type_=sa.Boolean()).over(
                partition_by=_window_partition_by,
                order_by=_window_order_by,
            ),
            sa.false(),
        )
