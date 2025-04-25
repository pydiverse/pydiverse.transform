from __future__ import annotations

from typing import Any
from uuid import UUID

import polars as pl
import sqlalchemy as sqa
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.sql.type_api import TypeEngine as TypeEngine

from pydiverse.common import Int64
from pydiverse.transform._internal.backend import sql
from pydiverse.transform._internal.backend.sql import SqlImpl
from pydiverse.transform._internal.backend.targets import Polars, Target
from pydiverse.transform._internal.ops import ops
from pydiverse.transform._internal.tree import types
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.col_expr import Cast, LiteralCol


class DuckDbImpl(SqlImpl):
    backend_name = "duckdb"

    @classmethod
    def export(
        cls,
        nd: AstNode,
        target: Target,
        *,
        schema_overrides: dict[UUID, Any],
    ):
        if isinstance(target, Polars):
            engine = sql.get_engine(nd)
            with engine.connect() as conn:
                return pl.read_database(
                    DuckDbImpl.build_query(nd),
                    connection=conn,
                    schema_overrides=schema_overrides,
                )
        return SqlImpl.export(nd, target, schema_overrides=schema_overrides)

    @classmethod
    def compile_cast(cls, cast: Cast, sqa_col: dict[str, sqa.Label]) -> Cast:
        if cast.val.dtype().is_float() and cast.target_type.is_int():
            return sqa.func.trunc(cls.compile_col_expr(cast.val, sqa_col)).cast(
                cls.sqa_type(cast.target_type)
            )
        return super().compile_cast(cast, sqa_col)

    @classmethod
    def compile_lit(cls, lit: LiteralCol) -> sqa.ColumnElement:
        if types.without_const(lit.dtype()) == Int64():
            return sqa.cast(lit.val, sqa.BigInteger)
        return super().compile_lit(lit)

    @classmethod
    def fix_fn_types(
        cls, fn: sql.ColFn, val: sqa.ColumnElement, *args: sqa.ColumnElement
    ) -> sqa.ColumnElement:
        if fn.op == ops.sum:
            return sqa.cast(val, type_=args[0].type)
        return val

    @classmethod
    def compile_ordered_aggregation(
        cls, *args: sqa.ColumnElement, order_by: list[sqa.UnaryExpression], impl
    ) -> sqa.ColumnElement:
        return impl(*args[:-1], aggregate_order_by(args[-1], *order_by))


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

    @impl(ops.str_join)
    def _str_join(x, delim):
        return sqa.func.string_agg(x, delim)

    @impl(ops.list_agg)
    def _list_agg(x):
        return sqa.func.array_agg(x)
