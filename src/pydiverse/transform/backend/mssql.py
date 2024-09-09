from __future__ import annotations

from typing import Any

import sqlalchemy as sqa

from pydiverse.transform import ops
from pydiverse.transform.backend import sql
from pydiverse.transform.backend.sql import SqlImpl
from pydiverse.transform.tree import dtypes, verbs
from pydiverse.transform.tree.col_expr import (
    CaseExpr,
    Cast,
    ColExpr,
    ColFn,
    ColName,
    LiteralCol,
    Order,
)
from pydiverse.transform.tree.registry import TypedOperatorImpl
from pydiverse.transform.tree.table_expr import TableExpr
from pydiverse.transform.util.warnings import warn_non_standard


class MsSqlImpl(SqlImpl):
    dialect_name = "mssql"

    @classmethod
    def build_select(cls, expr: TableExpr) -> Any:
        convert_table_bool_bit(expr)
        sql.create_aliases(expr)
        table, query = sql.compile_table_expr(expr)
        query.select = [
            (
                (Cast(col, dtypes.Bool()), name)
                if isinstance(col.dtype, dtypes.Bool)
                else (col, name)
            )
            for col, name in query.select
        ]
        return sql.compile_query(table, query)

    def _order_col(
        self, col: sqa.SQLColumnExpression, ordering
    ) -> list[sqa.SQLColumnExpression]:
        # MSSQL doesn't support nulls first / nulls last
        order_by_expressions = []

        # asc implies nulls first
        if not ordering.nulls_first and ordering.asc:
            order_by_expressions.append(sqa.func.iif(col.is_(None), 1, 0))

        # desc implies nulls last
        if ordering.nulls_first and not ordering.asc:
            order_by_expressions.append(sqa.func.iif(col.is_(None), 0, 1))

        order_by_expressions.append(col.asc() if ordering.asc else col.desc())
        return order_by_expressions


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
    if isinstance(expr, ColName):
        if isinstance(expr.dtype, dtypes.Bool):
            return expr == LiteralCol(1)
        return expr

    elif isinstance(expr, ColFn):
        op = MsSqlImpl.operator_registry.get_operator(expr.name)
        wants_bool_as_bit_input = not isinstance(
            op, ops.logical.BooleanBinary, ops.logical.Invert
        )

        converted = ColFn(
            expr.name,
            *(convert_col_bool_bit(arg, wants_bool_as_bit_input) for arg in expr.args),
            **{
                key: [convert_col_bool_bit(val, wants_bool_as_bit) for val in arr]
                for key, arr in expr.context_kwargs
            },
        )

        impl = MsSqlImpl.operator_registry.get_implementation(
            expr.name, tuple(arg.dtype for arg in expr.args)
        )
        returns_bool_as_bit = mssql_op_returns_bool_as_bit(impl)

        if wants_bool_as_bit and not returns_bool_as_bit:
            return CaseExpr([(converted, LiteralCol(1))], LiteralCol(0))
        elif not wants_bool_as_bit and returns_bool_as_bit:
            return converted == LiteralCol(1)

        return converted

    elif isinstance(expr, CaseExpr):
        return CaseExpr(
            [
                (
                    convert_col_bool_bit(cond, False),
                    convert_col_bool_bit(val, True),
                )
                for cond, val in expr.cases
            ],
            convert_col_bool_bit(expr.default_val, wants_bool_as_bit),
        )


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


def mssql_op_returns_bool_as_bit(implementation: TypedOperatorImpl) -> bool | None:
    if not dtypes.Bool().same_kind(implementation.return_type):
        return None

    # These operations return boolean types (not BIT)
    if isinstance(implementation.operator, ops.logical.Logical):
        return False

    return True


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
