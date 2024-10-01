from __future__ import annotations

import polars as pl
import sqlalchemy as sqa

from pydiverse.transform import ops
from pydiverse.transform.backend import sql
from pydiverse.transform.backend.sql import SqlImpl
from pydiverse.transform.backend.targets import Polars, Target
from pydiverse.transform.tree import dtypes
from pydiverse.transform.tree.ast import AstNode
from pydiverse.transform.tree.col_expr import Cast, Col


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

    @classmethod
    def compile_cast(cls, cast: Cast, sqa_col: dict[str, sqa.Label]) -> Cast:
        if cast.val.dtype() == dtypes.Float64 and cast.target_type == dtypes.Int64:
            return sqa.func.trunc(cls.compile_col_expr(cast.val, sqa_col)).cast(
                sqa.BigInteger()
            )
        return super().compile_cast(cast, sqa_col)


with DuckDbImpl.op(ops.FloorDiv()) as op:

    @op.auto
    def _floordiv(lhs, rhs):
        return sqa.func.divide(lhs, rhs)


with DuckDbImpl.op(ops.RFloorDiv()) as op:

    @op.auto
    def _floordiv(rhs, lhs):
        return sqa.func.divide(lhs, rhs)
