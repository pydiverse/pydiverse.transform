from __future__ import annotations

import sqlalchemy as sa

from pydiverse.transform.core import ops
from pydiverse.transform.core.util import OrderingDescriptor
from pydiverse.transform.lazy.sql_table.sql_table import SQLTableImpl


class MSSqlTableImpl(SQLTableImpl):
    _dialect_name = "mssql"

    def _order_col(
        self, col: sa.SQLColumnExpression, ordering: OrderingDescriptor
    ) -> list[sa.SQLColumnExpression]:
        # MSSQL doesn't support nulls first / nulls last
        order_by_expressions = []

        # asc implies nulls first
        if not ordering.nulls_first and ordering.asc:
            order_by_expressions.append(sa.func.iif(col.is_(None), 1, 0))

        # desc implies nulls last
        if ordering.nulls_first and not ordering.asc:
            order_by_expressions.append(sa.func.iif(col.is_(None), 0, 1))

        order_by_expressions.append(col.asc() if ordering.asc else col.desc())
        return order_by_expressions


with MSSqlTableImpl.op(ops.Mean()) as op:

    @op.auto
    def _mean(x):
        return sa.func.AVG(sa.cast(x, sa.Float()))
