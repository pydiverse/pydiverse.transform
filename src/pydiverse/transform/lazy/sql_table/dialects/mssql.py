from __future__ import annotations

import sqlalchemy as sa

from pydiverse.transform.core import Column, ops
from pydiverse.transform.core.table_impl import PostProcess, ProcessArg
from pydiverse.transform.lazy.sql_table.sql_table import SQLTableImpl


class MssqlTableImpl(SQLTableImpl):
    _dialect_name = "mssql"

    def post_process_order_by(self, col, o_by):
        col = col.asc() if o_by.asc else col.desc()
        # NULLS is not supported by TSQL, yet
        # col = col.nullsfirst() if o_by.nulls_first else col.nullslast()
        if o_by.asc and o_by.nulls_first is not None and not o_by.nulls_first:
            raise NotImplementedError(
                "NULLS LAST is not supported by TSQL on ascending order"
            )
        if not o_by.asc and o_by.nulls_first is not None and o_by.nulls_first:
            raise NotImplementedError(
                "NULLS FIRST is not supported by TSQL on descending order"
            )
        return col

    def get_literal_func(self, expr):
        def literal_func(*args, **kwargs):
            if isinstance(expr, bool):
                if expr:
                    return sa.literal(1) == sa.literal(1)
                else:
                    return sa.literal(0) == sa.literal(1)
            return sa.literal(expr)

        return literal_func

    def post_process_select_column(self, sql_col, pdt_col):
        if pdt_col.dtype == "bool" and not isinstance(pdt_col.expr, Column):
            return sa.cast(sa.func.iif(sql_col, 1, 0), sa.Boolean())
        return sql_col


with MssqlTableImpl.op(ops.Mean()) as op:

    @op.auto
    def _mean(x):
        return sa.func.AVG(sa.cast(x, sa.FLOAT))


with MssqlTableImpl.op(ops.Any()) as op:

    @op.auto
    def _any(x):
        # aggregation needs to be done as integer but result must be converted
        # back to boolean
        post_process = PostProcess(
            lambda _x: _x == 1, prevent_duplicate_key="mssql_to_bool"
        ) & ProcessArg(lambda _x: _x == 1)
        return sa.func.max(sa.func.iif(x, 1, 0)), post_process


with MssqlTableImpl.op(ops.All()) as op:

    @op.auto
    def _all(x):
        # aggregation needs to be done as integer but result must be converted
        # back to boolean
        post_process = PostProcess(
            lambda _x: _x == 1, prevent_duplicate_key="mssql_to_bool"
        ) & ProcessArg(lambda _x: _x == 1)
        return sa.func.min(sa.func.iif(x, 1, 0)), post_process
