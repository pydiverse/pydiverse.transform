from __future__ import annotations

import copy
import functools
from typing import Any

import sqlalchemy as sqa
from sqlalchemy.dialects.mssql import DATETIME2

from pydiverse.transform._internal import ops
from pydiverse.transform._internal.backend import sql
from pydiverse.transform._internal.backend.sql import SqlImpl
from pydiverse.transform._internal.errors import NotSupportedError
from pydiverse.transform._internal.tree import dtypes, verbs
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
from pydiverse.transform._internal.util.warnings import warn_non_standard


class MsSqlImpl(SqlImpl):
    @classmethod
    def inf():
        raise NotSupportedError("SQL Server does not support `inf`")

    @classmethod
    def nan():
        raise NotSupportedError("SQL Server does not support `nan`")

    @classmethod
    def build_select(cls, nd: AstNode, final_select: list[Col]) -> Any:
        # boolean / bit conversion
        for desc in nd.iter_subtree():
            if isinstance(desc, verbs.Verb):
                desc.map_col_roots(
                    functools.partial(
                        convert_bool_bit,
                        wants_bool_as_bit=not isinstance(
                            desc, verbs.Filter | verbs.Join
                        ),
                    )
                )

        # workaround for correct nulls_first / nulls_last behaviour on MSSQL
        for desc in nd.iter_subtree():
            if isinstance(desc, verbs.Arrange):
                desc.order_by = convert_order_list(desc.order_by)
            if isinstance(desc, verbs.Verb):
                for node in desc.iter_col_nodes():
                    if isinstance(node, ColFn) and (
                        arrange := node.context_kwargs.get("arrange")
                    ):
                        node.context_kwargs["arrange"] = convert_order_list(arrange)

        sql.create_aliases(nd, {})
        table, query, _ = cls.compile_ast(nd, {col._uuid: 1 for col in final_select})
        return cls.compile_query(table, query)

    @classmethod
    def sqa_type(cls, t: dtypes.Dtype):
        if isinstance(t, dtypes.DateTime):
            return DATETIME2

        return super().sqa_type(t)


def convert_order_list(order_list: list[Order]) -> list[Order]:
    new_list: list[Order] = []
    for ord in order_list:
        # is True / is False are important here since we don't want to do this costly
        # workaround if nulls_last is None (i.e. the user doesn't care)
        if ord.nulls_last is True and not ord.descending:
            new_list.append(
                Order(
                    CaseExpr([(ord.order_by.is_null(), LiteralCol(1))], LiteralCol(0)),
                )
            )

        elif ord.nulls_last is False and ord.descending:
            new_list.append(
                Order(
                    CaseExpr([(ord.order_by.is_null(), LiteralCol(0))], LiteralCol(1)),
                )
            )

        new_list.append(Order(ord.order_by, ord.descending, None))

    return new_list


# MSSQL doesn't have a boolean type. This means that expressions that return a boolean
# (e.g. ==, !=, >) can't be used in other expressions without casting to the BIT type.
# Conversely, after casting to BIT, we sometimes may need to convert back to booleans.


def convert_bool_bit(expr: ColExpr | Order, wants_bool_as_bit: bool) -> ColExpr | Order:
    if isinstance(expr, Order):
        return Order(
            convert_bool_bit(expr.order_by, wants_bool_as_bit),
            expr.descending,
            expr.nulls_last,
        )

    elif isinstance(expr, Col):
        if not wants_bool_as_bit and expr.dtype() == dtypes.Bool:
            return ColFn("__eq__", expr, LiteralCol(True))
        return expr

    elif isinstance(expr, ColFn):
        op = MsSqlImpl.registry.get_op(expr.name)
        wants_bool_as_bit_input = not isinstance(
            op, ops.logical.BooleanBinary | ops.logical.Invert
        )

        converted = copy.copy(expr)
        converted.args = [
            convert_bool_bit(arg, wants_bool_as_bit_input) for arg in expr.args
        ]
        converted.context_kwargs = {
            key: [convert_bool_bit(val, wants_bool_as_bit) for val in arr]
            for key, arr in expr.context_kwargs.items()
        }

        impl = MsSqlImpl.registry.get_impl(
            expr.name, tuple(arg.dtype() for arg in expr.args)
        )

        if isinstance(impl.return_type, dtypes.Bool):
            returns_bool_as_bit = not isinstance(op, ops.logical.Logical)

            if wants_bool_as_bit and not returns_bool_as_bit:
                return CaseExpr(
                    [(converted, LiteralCol(True)), (~converted, LiteralCol(False))],
                    None,
                )
            elif not wants_bool_as_bit and returns_bool_as_bit:
                return ColFn("__eq__", converted, LiteralCol(True))

        return converted

    elif isinstance(expr, CaseExpr):
        converted = copy.copy(expr)
        converted.cases = [
            (convert_bool_bit(cond, False), convert_bool_bit(val, True))
            for cond, val in expr.cases
        ]
        converted.default_val = (
            None
            if expr.default_val is None
            else convert_bool_bit(expr.default_val, wants_bool_as_bit)
        )

        return converted

    elif isinstance(expr, LiteralCol):
        return expr

    elif isinstance(expr, Cast):
        # TODO: does this really work for casting onto / from booleans? we probably have
        # to use wants_bool_as_bit in some way when casting to bool
        return Cast(
            convert_bool_bit(expr.val, wants_bool_as_bit=wants_bool_as_bit),
            expr.target_type,
        )

    raise AssertionError


with MsSqlImpl.op(ops.Equal()) as op:

    @op("str, str -> bool")
    def _eq(x, y):
        warn_non_standard(
            "MSSQL ignores trailing whitespace when comparing strings",
        )
        return x == y


with MsSqlImpl.op(ops.NotEqual()) as op:

    @op("str, str -> bool")
    def _ne(x, y):
        warn_non_standard(
            "MSSQL ignores trailing whitespace when comparing strings",
        )
        return x != y


with MsSqlImpl.op(ops.Less()) as op:

    @op("str, str -> bool")
    def _lt(x, y):
        warn_non_standard(
            "MSSQL ignores trailing whitespace when comparing strings",
        )
        return x < y


with MsSqlImpl.op(ops.LessEqual()) as op:

    @op("str, str -> bool")
    def _le(x, y):
        warn_non_standard(
            "MSSQL ignores trailing whitespace when comparing strings",
        )
        return x <= y


with MsSqlImpl.op(ops.Greater()) as op:

    @op("str, str -> bool")
    def _gt(x, y):
        warn_non_standard(
            "MSSQL ignores trailing whitespace when comparing strings",
        )
        return x > y


with MsSqlImpl.op(ops.GreaterEqual()) as op:

    @op("str, str -> bool")
    def _ge(x, y):
        warn_non_standard(
            "MSSQL ignores trailing whitespace when comparing strings",
        )
        return x >= y


with MsSqlImpl.op(ops.Pow()) as op:

    @op.auto
    def _pow(lhs, rhs):
        # In MSSQL, the output type of pow is the same as the input type.
        # This means, that if lhs is a decimal, then we may very easily loose
        # a lot of precision if the exponent is <= 1
        # https://learn.microsoft.com/en-us/sql/t-sql/functions/power-transact-sql?view=sql-server-ver16
        return sqa.func.POWER(sqa.cast(lhs, sqa.Double()), rhs, type_=sqa.Double())


with MsSqlImpl.op(ops.RPow()) as op:

    @op.auto
    def _rpow(rhs, lhs):
        return _pow(lhs, rhs)


with MsSqlImpl.op(ops.StrLen()) as op:

    @op.auto
    def _str_length(x):
        return sqa.func.LENGTH(x + "a", type_=sqa.Integer()) - 1


with MsSqlImpl.op(ops.StrReplaceAll()) as op:

    @op.auto
    def _replace_all(x, y, z):
        x = x.collate("Latin1_General_CS_AS")
        return sqa.func.REPLACE(x, y, z, type_=x.type)


with MsSqlImpl.op(ops.StrStartsWith()) as op:

    @op.auto
    def _startswith(x, y):
        x = x.collate("Latin1_General_CS_AS")
        return x.startswith(y, autoescape=True)


with MsSqlImpl.op(ops.StrEndsWith()) as op:

    @op.auto
    def _endswith(x, y):
        x = x.collate("Latin1_General_CS_AS")
        return x.endswith(y, autoescape=True)


with MsSqlImpl.op(ops.StrContains()) as op:

    @op.auto
    def _contains(x, y):
        x = x.collate("Latin1_General_CS_AS")
        return x.contains(y, autoescape=True)


with MsSqlImpl.op(ops.StrSlice()) as op:

    @op.auto
    def _str_slice(x, offset, length):
        return sqa.func.SUBSTRING(x, offset + 1, length)


with MsSqlImpl.op(ops.DtDayOfWeek()) as op:

    @op.auto
    def _day_of_week(x):
        # Offset DOW such that Mon=1, Sun=7
        _1 = sqa.literal_column("1")
        _2 = sqa.literal_column("2")
        _7 = sqa.literal_column("7")
        return (sqa.extract("dow", x) + sqa.text("@@DATEFIRST") - _2) % _7 + _1


with MsSqlImpl.op(ops.Mean()) as op:

    @op.auto
    def _mean(x):
        return sqa.func.AVG(sqa.cast(x, sqa.Double()), type_=sqa.Double())


with MsSqlImpl.op(ops.Log()) as op:

    @op.auto
    def _log(x):
        return sqa.func.log(x)


with MsSqlImpl.op(ops.Ceil()) as op:

    @op.auto
    def _ceil(x):
        return sqa.func.ceiling(x)


with MsSqlImpl.op(ops.StrToDateTime()) as op:

    @op.auto
    def _str_to_datetime(x):
        return sqa.cast(x, DATETIME2)


with MsSqlImpl.op(ops.IsInf()) as op:

    @op.auto
    def _is_inf(x):
        return False


with MsSqlImpl.op(ops.IsNotInf()) as op:

    @op.auto
    def _is_not_inf(x):
        return True


with MsSqlImpl.op(ops.IsNan()) as op:

    @op.auto
    def _is_nan(x):
        return False


with MsSqlImpl.op(ops.IsNotNan()) as op:

    @op.auto
    def _is_not_nan(x):
        return True
