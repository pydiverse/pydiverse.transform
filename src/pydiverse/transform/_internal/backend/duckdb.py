from __future__ import annotations

from typing import Any

import polars as pl
import sqlalchemy as sqa
from sqlalchemy.sql.type_api import TypeEngine as TypeEngine

from pydiverse.transform._internal.backend import sql
from pydiverse.transform._internal.backend.sql import SqlImpl
from pydiverse.transform._internal.backend.targets import Polars, Target
from pydiverse.transform._internal.ops import ops
from pydiverse.transform._internal.tree import types, verbs
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.col_expr import Cast, Col, ColFn, LiteralCol


class DuckDbImpl(SqlImpl):
    @classmethod
    def export(
        cls,
        nd: AstNode,
        target: Target,
        final_select: list[Col],
        schema_overrides: dict[str, Any],
    ):
        # insert casts after sum() over integer columns (duckdb converts them to floats)
        for desc in nd.iter_subtree():
            if isinstance(desc, verbs.Verb):
                desc.map_col_nodes(
                    lambda u: Cast(u, types.Int64())
                    if isinstance(u, ColFn)
                    and u.name == "sum"
                    and u.dtype() == types.Int64
                    else u
                )

        if isinstance(target, Polars):
            engine = sql.get_engine(nd)
            with engine.connect() as conn:
                return pl.read_database(
                    DuckDbImpl.build_query(nd, final_select),
                    connection=conn,
                    schema_overrides=schema_overrides,
                )
        return SqlImpl.export(nd, target, final_select, schema_overrides)

    @classmethod
    def compile_cast(cls, cast: Cast, sqa_col: dict[str, sqa.Label]) -> Cast:
        if cast.val.dtype() == types.Float64 and cast.target_type == types.Int64:
            return sqa.func.trunc(cls.compile_col_expr(cast.val, sqa_col)).cast(
                sqa.BigInteger()
            )
        return super().compile_cast(cast, sqa_col)

    @classmethod
    def compile_lit(cls, lit: LiteralCol) -> sqa.ColumnElement:
        if lit.dtype() == types.Int64:
            return sqa.cast(lit.val, sqa.BigInteger)
        return super().compile_lit(lit)


with DuckDbImpl.impl_store.impl_manager as impl:

    @impl(ops.floordiv)
    def _floordiv(lhs, rhs):
        return sqa.func.divide(lhs, rhs)

    @impl(ops.is_inf)
    def _is_inf(x):
        return sqa.func.isinf(x)

    @impl(ops.is_not_inf)
    def _is_not_inf(x):
        return sqa.func.isfinite(x)

    @impl(ops.is_nan)
    def _is_nan(x):
        return sqa.func.isnan(x)

    @impl(ops.is_not_nan)
    def _is_not_nan(x):
        return ~sqa.func.isnan(x)
