from __future__ import annotations

import copy
from typing import Any

import sqlalchemy as sqa

from pydiverse.transform import ops
from pydiverse.transform.backend import sql
from pydiverse.transform.backend.sql import SqlImpl
from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree import dtypes, verbs
from pydiverse.transform.tree.col_expr import (
    CaseExpr,
    ColExpr,
    ColFn,
    ColName,
    LiteralCol,
    Order,
)
from pydiverse.transform.tree.table_expr import TableExpr
from pydiverse.transform.util.warnings import warn_non_standard


class MsSqlImpl(SqlImpl):
    dialect_name = "mssql"

    @classmethod
    def build_select(cls, expr: TableExpr) -> Any:
        convert_table_bool_bit(expr)
        set_nulls_position_table(expr)
        sql.create_aliases(expr, {})
        table, query, _ = sql.compile_table_expr(expr)
        return sql.compile_query(table, query)


def convert_order_list(order_list: list[Order]) -> list[Order]:
    new_list = []
    for ord in order_list:
        new_list.append(Order(ord.order_by, ord.descending, None))
        # is True / is False are important here since we don't want to do this costly
        # workaround if nulls_last is None (i.e. the user doesn't care)
        if ord.nulls_last is True and not ord.descending:
            new_list.append(
                Order(
                    CaseExpr([(ord.order_by.is_null(), LiteralCol(1))], LiteralCol(0)),
                    False,
                    None,
                )
            )
        elif ord.nulls_last is False and ord.descending:
            new_list.append(
                Order(
                    CaseExpr([(ord.order_by.is_null(), LiteralCol(0))], LiteralCol(1)),
                    True,
                    None,
                )
            )
    return new_list


def set_nulls_position_table(expr: TableExpr):
    if isinstance(expr, verbs.UnaryVerb):
        set_nulls_position_table(expr.table)
        for col in expr.col_exprs():
            set_nulls_position_col(col)

        if isinstance(expr, verbs.Arrange):
            expr.order_by = convert_order_list(expr.order_by)

    elif isinstance(expr, verbs.Join):
        set_nulls_position_table(expr.left)
        set_nulls_position_table(expr.right)


def set_nulls_position_col(expr: ColExpr):
    if isinstance(expr, ColFn):
        for arg in expr.args:
            set_nulls_position_col(arg)
        if arr := expr.context_kwargs.get("arrange"):
            expr.context_kwargs["arrange"] = convert_order_list(arr)

    elif isinstance(expr, CaseExpr):
        set_nulls_position_col(expr.default_val)
        for cond, val in expr.cases:
            set_nulls_position_col(cond)
            set_nulls_position_col(val)


# Boolean / Bit Conversion
#
# MSSQL doesn't have a boolean type. This means that expressions that
# return a boolean (e.g. ==, !=, >) can't be used in other expressions
# without casting to the BIT type.
# Conversely, after casting to BIT, we sometimes may need to convert
# back to booleans.


def convert_col_bool_bit(
    expr: ColExpr | Order, wants_bool_as_bit: bool
) -> ColExpr | Order:
    if isinstance(expr, Order):
        return Order(
            convert_col_bool_bit(expr.order_by), expr.descending, expr.nulls_last
        )

    elif isinstance(expr, ColName):
        if isinstance(expr.dtype, dtypes.Bool):
            return ColFn("__eq__", expr, LiteralCol(1), dtype=dtypes.Bool())
        return expr

    elif isinstance(expr, ColFn):
        op = MsSqlImpl.operator_registry.get_operator(expr.name)
        wants_bool_as_bit_input = not isinstance(
            op, (ops.logical.BooleanBinary, ops.logical.Invert)
        )

        converted = copy.copy(expr)
        converted.args = [
            convert_col_bool_bit(arg, wants_bool_as_bit_input) for arg in expr.args
        ]
        converted.context_kwargs = {
            key: [convert_col_bool_bit(val, wants_bool_as_bit) for val in arr]
            for key, arr in expr.context_kwargs
        }

        impl = MsSqlImpl.operator_registry.get_implementation(
            expr.name, tuple(arg.dtype for arg in expr.args)
        )

        if isinstance(impl.return_type, dtypes.Bool):
            returns_bool_as_bit = not isinstance(op, ops.logical.Logical)

            if wants_bool_as_bit and not returns_bool_as_bit:
                return CaseExpr([(converted, LiteralCol(1))], LiteralCol(0))
            elif not wants_bool_as_bit and returns_bool_as_bit:
                return ColFn("__eq__", converted, LiteralCol(1), dtype=dtypes.Bool())

        return converted

    elif isinstance(expr, CaseExpr):
        converted = copy.copy(expr)
        converted.cases = [
            (
                convert_col_bool_bit(cond, False),
                convert_col_bool_bit(val, True),
            )
            for cond, val in expr.cases
        ]
        converted.default_val = convert_col_bool_bit(
            expr.default_val, wants_bool_as_bit
        )
        return converted

    elif isinstance(expr, LiteralCol):
        return expr

    raise AssertionError


def convert_table_bool_bit(expr: TableExpr):
    if isinstance(expr, verbs.UnaryVerb):
        convert_table_bool_bit(expr.table)
        expr.replace_col_exprs(
            lambda col: convert_col_bool_bit(col, not isinstance(expr, verbs.Filter))
        )

    elif isinstance(expr, verbs.Join):
        convert_table_bool_bit(expr.left)
        convert_table_bool_bit(expr.right)
        expr.on = convert_col_bool_bit(expr.on, False)

    else:
        assert isinstance(expr, Table)


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
        warn_non_standard(
            "MSSQL ignores trailing whitespace when computing string length",
        )
        return sqa.func.LENGTH(x, type_=sqa.Integer())


with MsSqlImpl.op(ops.StrReplaceAll()) as op:

    @op.auto
    def _replace(x, y, z):
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
