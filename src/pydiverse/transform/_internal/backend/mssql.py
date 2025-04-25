from __future__ import annotations

import copy
import functools
from typing import Any, Literal
from uuid import UUID

import sqlalchemy as sqa
from sqlalchemy.dialects.mssql import BIT, DATETIME2, TINYINT

from pydiverse.common import Bool, Int, String
from pydiverse.transform._internal.backend import sql
from pydiverse.transform._internal.backend.sql import SqlImpl
from pydiverse.transform._internal.backend.targets import Target
from pydiverse.transform._internal.errors import NotSupportedError
from pydiverse.transform._internal.ops import ops
from pydiverse.transform._internal.pipe.table import Cache
from pydiverse.transform._internal.tree import types, verbs
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.col_expr import (
    CaseExpr,
    Cast,
    Col,
    ColExpr,
    ColFn,
    LiteralCol,
    Order,
)


class MsSqlImpl(SqlImpl):
    backend_name = "mssql"

    @classmethod
    def inf(cls):
        raise NotSupportedError("SQL Server does not support `inf`")

    @classmethod
    def nan(cls):
        raise NotSupportedError("SQL Server does not support `nan`")

    @classmethod
    def default_collation(cls):
        return "Latin1_General_bin"

    @classmethod
    def export(
        cls,
        nd: AstNode,
        target: Target,
        *,
        schema_overrides: dict[UUID, Any],
    ) -> Any:
        final_select = Cache.from_ast(nd).selected_cols()

        for col in final_select:
            if types.without_const(col.dtype()) == Bool():
                if col._uuid not in schema_overrides:
                    schema_overrides[col._uuid] = col.dtype().to_polars()

        return super().export(nd, target, schema_overrides=schema_overrides)

    @classmethod
    def build_select(cls, nd: AstNode, *, final_select: list[Col] | None = None) -> Any:
        if final_select is None:
            final_select = Cache.from_ast(nd).selected_cols()

        # boolean / bit conversion
        for desc in nd.iter_subtree_postorder():
            if isinstance(desc, verbs.Verb):
                desc.map_col_roots(
                    functools.partial(
                        convert_bool_bit,
                        desired_return_type="bool"
                        if isinstance(desc, verbs.Filter | verbs.Join)
                        else "bit",
                    )
                )

        # workaround for correct nulls_first / nulls_last behaviour on MSSQL
        for desc in nd.iter_subtree_postorder():
            if isinstance(desc, verbs.Arrange):
                desc.order_by = convert_order_list(desc.order_by)
            if isinstance(desc, verbs.Verb):
                for node in desc.iter_col_nodes():
                    if isinstance(node, ColFn) and (
                        arrange := node.context_kwargs.get("arrange")
                    ):
                        node.context_kwargs["arrange"] = convert_order_list(arrange)

        sql.create_aliases(nd, {})
        table, query, sqa_expr = cls.compile_ast(
            nd, {col._uuid: 1 for col in final_select}
        )
        return cls.compile_query(table, query, sqa_expr)

    @classmethod
    def compile_ordered_aggregation(
        cls, *args: sqa.ColumnElement, order_by: list[sqa.UnaryExpression], impl
    ):
        return impl(*args).within_group(*order_by)

    @classmethod
    def fix_fn_types(
        cls, fn: ColFn, val: sqa.ColumnElement, *args: sqa.ColumnElement
    ) -> sqa.ColumnElement:
        if (
            fn.op in (ops.any, ops.all, ops.min, ops.max)
            and types.without_const(fn.dtype()) == Bool()
        ):
            return val.cast(BIT)

        return val


def convert_order_list(order_list: list[Order]) -> list[Order]:
    new_list: list[Order] = []
    for ord in order_list:
        # is True / is False are important here since we don't want to do this costly
        # workaround if nulls_last is None (i.e. the user doesn't care)
        if ord.nulls_last is not None and (ord.nulls_last ^ ord.descending):
            new_list.append(
                Order(
                    CaseExpr(
                        [(ord.order_by.is_null(), LiteralCol(int(ord.nulls_last)))],
                        LiteralCol(int(not ord.nulls_last)),
                    )
                )
            )

        new_list.append(Order(ord.order_by, ord.descending, None))

    return new_list


# MSSQL doesn't have a boolean type. This means that expressions that return a boolean
# (e.g. ==, !=, >) can't be used in other expressions without casting to the BIT type.
# Conversely, after casting to BIT, we sometimes may need to convert back to booleans.
# For any / all, we need to use max / min, which can't be applied to BIT either. Thus we
# convert to int in this case.


def convert_bool_bit(
    expr: ColExpr | Order,
    desired_return_type: Literal["bool", "bit"],
) -> ColExpr | Order:
    if isinstance(expr, Order):
        return Order(
            convert_bool_bit(expr.order_by, "bit"),
            expr.descending,
            expr.nulls_last,
        )

    result = expr

    if isinstance(expr, Col):
        return_type = "bit"

    elif isinstance(expr, ColFn):
        desired_arg_type = (
            "bool"
            if expr.op
            in (
                ops.bool_and,
                ops.bool_or,
                ops.bool_invert,
                ops.horizontal_all,
                ops.horizontal_any,
            )
            else "bit"
        )

        result = copy.copy(expr)
        result.args = list(convert_bool_bit(arg, desired_arg_type) for arg in expr.args)
        result.context_kwargs = {
            key: [convert_bool_bit(val, "bit") for val in arr]
            for key, arr in expr.context_kwargs.items()
        }

        return_type = (
            "bool"
            if expr.op
            in (
                ops.bool_and,
                ops.bool_or,
                ops.bool_xor,
                ops.bool_invert,
                ops.equal,
                ops.not_equal,
                ops.greater_equal,
                ops.greater_than,
                ops.less_equal,
                ops.less_than,
                ops.is_null,
                ops.is_not_null,
                ops.is_in,
                ops.str_starts_with,
                ops.str_ends_with,
                ops.str_contains,
                ops.horizontal_any,
                ops.horizontal_all,
            )
            else "bit"
        )

    elif isinstance(expr, CaseExpr):
        return_type = "bit"
        result = copy.copy(expr)
        result.cases = [
            (convert_bool_bit(cond, "bool"), convert_bool_bit(val, "bit"))
            for cond, val in expr.cases
        ]
        result.default_val = (
            None
            if expr.default_val is None
            else convert_bool_bit(expr.default_val, "bit")
        )

    elif isinstance(expr, LiteralCol):
        return_type = "bit"

    elif isinstance(expr, Cast):
        return Cast(
            convert_bool_bit(expr.val, "bit"),
            expr.target_type,
        )

    if types.without_const(expr.dtype()) == Bool():
        if desired_return_type == "bool" and return_type == "bit":
            return ColFn(ops.equal, result, LiteralCol(True))

        if return_type == "bool" and desired_return_type == "bit":
            return CaseExpr(
                [(result, LiteralCol(True)), (~result, LiteralCol(False))],
                None,
            )

    return result


with MsSqlImpl.impl_store.impl_manager as impl:

    @impl(ops.equal, String(), String())
    def _eq(x, y):
        return (sqa.func.LENGTH(x + "a") == sqa.func.LENGTH(y + "a")) & (
            x.collate("Latin1_General_bin") == y
        )

    @impl(ops.not_equal, String(), String())
    def _ne(x, y):
        return (sqa.func.LENGTH(x + "a") != sqa.func.LENGTH(y + "a")) | (
            x.collate("Latin1_General_bin") != y
        )

    @impl(ops.less_than, String(), String())
    def _lt(x, y):
        y_ = sqa.func.SUBSTRING(y, 1, sqa.func.LENGTH(x + "a") - 1)
        return (x.collate("Latin1_General_bin") < y_) | (
            (sqa.func.LENGTH(x + "a") < sqa.func.LENGTH(y + "a"))
            & (x.collate("Latin1_General_bin") == y_)
        )

    @impl(ops.less_equal, String(), String())
    def _le(x, y):
        y_ = sqa.func.SUBSTRING(y, 1, sqa.func.LENGTH(x + "a") - 1)
        return (x.collate("Latin1_General_bin") < y_) | (
            (sqa.func.LENGTH(x + "a") <= sqa.func.LENGTH(y + "a"))
            & (x.collate("Latin1_General_bin") == y_)
        )

    @impl(ops.greater_than, String(), String())
    def _gt(x, y):
        y_ = sqa.func.SUBSTRING(y, 1, sqa.func.LENGTH(x + "a") - 1)
        return (x.collate("Latin1_General_bin") > y_) | (
            (sqa.func.LENGTH(x + "a") > sqa.func.LENGTH(y + "a"))
            & (x.collate("Latin1_General_bin") == y_)
        )

    @impl(ops.greater_equal, String(), String())
    def _ge(x, y):
        y_ = sqa.func.SUBSTRING(y, 1, sqa.func.LENGTH(x + "a") - 1)
        return (x.collate("Latin1_General_bin") > y_) | (
            (sqa.func.LENGTH(x + "a") >= sqa.func.LENGTH(y + "a"))
            & (x.collate("Latin1_General_bin") == y_)
        )

    @impl(ops.str_len)
    def _str_length(x):
        return sqa.func.LENGTH(x + "a", type_=sqa.BigInteger()) - 1

    @impl(ops.str_replace_all)
    def _str_replace_all(x, y, z):
        x = x.collate("Latin1_General_CS_AS")
        return sqa.func.REPLACE(x, y, z, type_=x.type)

    @impl(ops.str_starts_with)
    def _str_starts_with(x, y):
        x = x.collate("Latin1_General_CS_AS")
        return x.startswith(y, autoescape=True)

    @impl(ops.str_ends_with)
    def _str_ends_with(x, y):
        x = x.collate("Latin1_General_CS_AS")
        return x.endswith(y, autoescape=True)

    @impl(ops.str_contains)
    def _contains(x, y):
        x = x.collate("Latin1_General_CS_AS")
        return x.contains(y, autoescape=True)

    @impl(ops.str_slice)
    def _str_slice(x, offset, length):
        return sqa.func.SUBSTRING(x, offset + 1, length)

    @impl(ops.dt_day_of_week)
    def _dt_day_of_week(x):
        # Offset DOW such that Mon=1, Sun=7
        _1 = sqa.literal_column("1")
        _2 = sqa.literal_column("2")
        _7 = sqa.literal_column("7")
        return (sqa.extract("dow", x) + sqa.text("@@DATEFIRST") - _2) % _7 + _1

    @impl(ops.mean)
    def _mean(x):
        return sqa.func.AVG(sqa.cast(x, sqa.Double()), type_=sqa.Double())

    @impl(ops.log)
    def _log(x):
        return sqa.func.log(x)

    @impl(ops.ceil)
    def _ceil(x):
        return sqa.func.ceiling(x)

    @impl(ops.str_to_datetime)
    def _str_to_datetime(x):
        return sqa.cast(x, DATETIME2)

    @impl(ops.is_inf)
    def _is_inf(x):
        return False

    @impl(ops.is_not_inf)
    def _is_not_inf(x):
        return True

    @impl(ops.is_nan)
    def _is_nan(x):
        return False

    @impl(ops.is_not_nan)
    def _is_not_nan(x):
        return True

    @impl(ops.pow)
    def _pow(x, y):
        return_type = sqa.Double()
        if isinstance(x.type, sqa.Numeric) and isinstance(y.type, sqa.Numeric):
            return_type = sqa.Numeric()
        return sqa.func.POWER(x, y, type_=return_type)

    @impl(ops.pow, Int(), Int())
    def _pow_int(x, y):
        return sqa.func.POWER(sqa.cast(x, type_=sqa.Double()), y)

    @impl(ops.all)
    @impl(ops.min, Bool())
    def _all(x):
        return sqa.func.min(sqa.cast(x, TINYINT))

    @impl(ops.any)
    @impl(ops.max, Bool())
    def _all(x):
        return sqa.func.max(sqa.cast(x, TINYINT))
