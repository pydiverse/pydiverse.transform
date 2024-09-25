from __future__ import annotations

import polars as pl

from pydiverse.transform.backend import sql
from pydiverse.transform.backend.sql import SqlImpl
from pydiverse.transform.backend.targets import Polars, Target
from pydiverse.transform.tree.ast import AstNode
from pydiverse.transform.tree.col_expr import Col


class DuckDbImpl(SqlImpl):
    dialect_name = "duckdb"

    @classmethod
    def export(cls, nd: AstNode, target: Target, final_select: list[Col]):
        if isinstance(target, Polars):
            engine = sql.get_engine(nd)
            with engine.connect() as conn:
                return pl.read_database(
                    DuckDbImpl.build_query(nd, final_select), connection=conn
                )
        return SqlImpl.export(nd, target, final_select)
