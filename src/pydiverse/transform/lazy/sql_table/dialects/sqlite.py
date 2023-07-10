from __future__ import annotations

import sqlalchemy as sa

from pydiverse.transform.core import ops
from pydiverse.transform.lazy.sql_table.sql_table import SQLTableImpl


class SQLiteTableImpl(SQLTableImpl):
    _dialect_name = "sqlite"


with SQLiteTableImpl.op(ops.Any()) as op:

    @op.auto
    def _any(x):
        return sa.func.max(x)


with SQLiteTableImpl.op(ops.All()) as op:

    @op.auto
    def _all(x):
        return sa.func.min(x)
