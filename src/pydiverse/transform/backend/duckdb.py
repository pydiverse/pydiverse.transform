from __future__ import annotations

import polars as pl
import sqlalchemy as sqa
from sqlalchemy.sql.type_api import TypeEngine as TypeEngine

from pydiverse.transform import ops
from pydiverse.transform.backend import sql
from pydiverse.transform.backend.sql import SqlImpl
from pydiverse.transform.backend.targets import Polars, Target
from pydiverse.transform.tree import dtypes, verbs
from pydiverse.transform.tree.ast import AstNode
from pydiverse.transform.tree.col_expr import Cast, Col, ColFn, LiteralCol


class DuckDbImpl(SqlImpl):
    dialect_name = "duckdb"

    @classmethod
    def export(cls, nd: AstNode, target: Target, final_select: list[Col]):
        # insert casts after sum() over integer columns (duckdb converts them to floats)
        for desc in nd.iter_subtree():
            if isinstance(desc, verbs.Verb):
                desc.map_col_nodes(
                    lambda u: Cast(u, dtypes.Int64())
                    if isinstance(u, ColFn)
                    and u.name == "sum"
                    and u.dtype() == dtypes.Int64
                    else u
                )

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

    @classmethod
    def compile_lit(cls, lit: LiteralCol) -> sqa.ColumnElement:
        if lit.dtype() == dtypes.Int64:
            return sqa.cast(lit.val, sqa.BigInteger)
        return super().compile_lit(lit)


with DuckDbImpl.op(ops.FloorDiv()) as op:

    @op.auto
    def _floordiv(lhs, rhs):
        return sqa.func.divide(lhs, rhs)


with DuckDbImpl.op(ops.RFloorDiv()) as op:

    @op.auto
    def _floordiv(rhs, lhs):
        return sqa.func.divide(lhs, rhs)


with DuckDbImpl.op(ops.IsInf()) as op:

    @op.auto
    def _is_inf(x):
        return sqa.func.isinf(x)


with DuckDbImpl.op(ops.IsNotInf()) as op:

    @op.auto
    def _is_not_inf(x):
        return sqa.func.isfinite(x)


with DuckDbImpl.op(ops.IsNan()) as op:

    @op.auto
    def _is_nan(x):
        return sqa.func.isnan(x)


with DuckDbImpl.op(ops.IsNotNan()) as op:

    @op.auto
    def _is_not_nan(x):
        return ~sqa.func.isnan(x)
