from __future__ import annotations

import polars as pl

from pydiverse.transform.backend import sql
from pydiverse.transform.backend.sql import SqlImpl
from pydiverse.transform.backend.targets import Polars, Target
from pydiverse.transform.tree.table_expr import TableExpr


class DuckDbImpl(SqlImpl):
    dialect_name = "duckdb"

    @classmethod
    def export(cls, expr: TableExpr, target: Target):
        if isinstance(target, Polars):
            engine = sql.get_engine(expr)
            with engine.connect() as conn:
                return pl.read_database(DuckDbImpl.build_query(expr), connection=conn)
        return SqlImpl.export(expr, target)
