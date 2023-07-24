from __future__ import annotations

import sqlalchemy as sa

from pydiverse.transform import ops
from pydiverse.transform.lazy.sql_table.sql_table import SQLTableImpl


class PostgresTableImpl(SQLTableImpl):
    _dialect_name = "postgresql"


with PostgresTableImpl.op(ops.Any()) as op:

    @op.auto
    def _any(x):
        return sa.func.coalesce(sa.func.BOOL_OR(x), False)


with PostgresTableImpl.op(ops.All()) as op:

    @op.auto
    def _all(x):
        return sa.func.coalesce(sa.func.BOOL_AND(x), False)
