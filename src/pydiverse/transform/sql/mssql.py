from __future__ import annotations

import sqlalchemy as sa

from pydiverse.transform import ops
from pydiverse.transform._typing import CallableT
from pydiverse.transform.core import dtypes
from pydiverse.transform.core.expressions import TypedValue
from pydiverse.transform.core.expressions.expressions import Column
from pydiverse.transform.core.registry import TypedOperatorImpl
from pydiverse.transform.core.util import OrderingDescriptor
from pydiverse.transform.ops import Operator, OPType
from pydiverse.transform.sql.sql_table import SQLTableImpl
from pydiverse.transform.util.warnings import warn_non_standard


class MSSqlTableImpl(SQLTableImpl):
    _dialect_name = "mssql"

    def _build_select_select(self, select):
        s = []
        for name, uuid_ in self.selected_cols():
            sql_col = self.cols[uuid_].compiled(self.sql_columns)
            if not isinstance(sql_col, sa.sql.ColumnElement):
                sql_col = sa.literal(sql_col)
            if dtypes.Bool().same_kind(self.cols[uuid_].dtype):
                # Make sure that any boolean values get stored as bit
                sql_col = sa.cast(sql_col, sa.Boolean())
            s.append(sql_col.label(name))
        return select.with_only_columns(*s)

    def _order_col(
        self, col: sa.SQLColumnExpression, ordering: OrderingDescriptor
    ) -> list[sa.SQLColumnExpression]:
        # MSSQL doesn't support nulls first / nulls last
        order_by_expressions = []

        # asc implies nulls first
        if not ordering.nulls_first and ordering.asc:
            order_by_expressions.append(sa.func.iif(col.is_(None), 1, 0))

        # desc implies nulls last
        if ordering.nulls_first and not ordering.asc:
            order_by_expressions.append(sa.func.iif(col.is_(None), 0, 1))

        order_by_expressions.append(col.asc() if ordering.asc else col.desc())
        return order_by_expressions

    class ExpressionCompiler(SQLTableImpl.ExpressionCompiler):
        def translate(self, expr, **kwargs):
            mssql_bool_as_bit = True
            if verb := kwargs.get("verb"):
                mssql_bool_as_bit = verb not in ("filter", "join")

            return super().translate(
                expr, **kwargs, mssql_bool_as_bit=mssql_bool_as_bit
            )

        def _translate(self, expr, **kwargs):
            if context := kwargs.get("context"):
                if context == "case_val":
                    kwargs["mssql_bool_as_bit"] = True
                elif context == "case_cond":
                    kwargs["mssql_bool_as_bit"] = False

            return super()._translate(expr, **kwargs)

        def _translate_col(self, col: Column, **kwargs):
            # If mssql_bool_as_bit is true, then we can just return the
            # precompiled col. Otherwise, we must recompile it to ensure
            # we return booleans as bools and not as bits.
            if kwargs.get("mssql_bool_as_bit") is True:
                return super()._translate_col(col, **kwargs)

            # Can either be a base SQL column, or a reference to an expression
            if col.uuid in self.backend.sql_columns:
                is_bool = dtypes.Bool().same_kind(self.backend.cols[col.uuid].dtype)

                def sql_col(cols, **kw):
                    sql_col = cols[col.uuid]
                    if is_bool:
                        return mssql_convert_bit_to_bool(sql_col)
                    return sql_col

                return TypedValue(sql_col, col.dtype, OPType.EWISE)

            meta_data = self.backend.cols[col.uuid]
            return self._translate(meta_data.expr, **kwargs)

        def _translate_function_value(
            self, implementation, op_args, context_kwargs, *, verb=None, **kwargs
        ):
            value = super()._translate_function_value(
                implementation,
                op_args,
                context_kwargs,
                verb=verb,
                **kwargs,
            )

            bool_as_bit = kwargs.get("mssql_bool_as_bit")
            returns_bool_as_bit = mssql_op_returns_bool_as_bit(implementation)
            return mssql_convert_bool_bit_value(value, bool_as_bit, returns_bool_as_bit)

        def _translate_function_arguments(self, expr, operator, **kwargs):
            kwargs["mssql_bool_as_bit"] = mssql_op_wants_bool_as_bit(operator)
            return super()._translate_function_arguments(expr, operator, **kwargs)


# Boolean / Bit Conversion
#
# MSSQL doesn't have a boolean type. This means that expressions that
# return a boolean (e.g. ==, !=, >) can't be used in other expressions
# without casting to the BIT type.
# Conversely, after casting to BIT, we sometimes may need to convert
# back to booleans.


def mssql_op_wants_bool_as_bit(operator: Operator) -> bool:
    # These operations want boolean types (not BIT) as input
    exceptions = [
        ops.logical.BooleanBinary,
        ops.logical.Invert,
    ]

    for exception in exceptions:
        if isinstance(operator, exception):
            return False

    return True


def mssql_op_returns_bool_as_bit(implementation: TypedOperatorImpl) -> bool | None:
    if not dtypes.Bool().same_kind(implementation.rtype):
        return None

    # These operations return boolean types (not BIT)
    if isinstance(implementation.operator, ops.logical.Logical):
        return False

    return True


def mssql_convert_bit_to_bool(x: sa.SQLColumnExpression):
    return x == sa.literal_column("1")


def mssql_convert_bool_to_bit(x: sa.SQLColumnExpression):
    return sa.case(
        (x, sa.literal_column("1")),
        (sa.not_(x), sa.literal_column("0")),
    )


def mssql_convert_bool_bit_value(
    value_func: CallableT,
    wants_bool_as_bit: bool | None,
    is_bool_as_bit: bool | None,
) -> CallableT:
    if wants_bool_as_bit is True and is_bool_as_bit is False:

        def value(*args, **kwargs):
            x = value_func(*args, **kwargs)
            return mssql_convert_bool_to_bit(x)

        return value

    if wants_bool_as_bit is False and is_bool_as_bit is True:

        def value(*args, **kwargs):
            x = value_func(*args, **kwargs)
            return mssql_convert_bit_to_bool(x)

        return value

    return value_func


# Operators


with MSSqlTableImpl.op(ops.Equal()) as op:

    @op("str, str -> bool")
    def _eq(x, y):
        warn_non_standard(
            "MSSQL ignores trailing whitespace when comparing strings",
        )
        return x == y


with MSSqlTableImpl.op(ops.NotEqual()) as op:

    @op("str, str -> bool")
    def _ne(x, y):
        warn_non_standard(
            "MSSQL ignores trailing whitespace when comparing strings",
        )
        return x != y


with MSSqlTableImpl.op(ops.Less()) as op:

    @op("str, str -> bool")
    def _lt(x, y):
        warn_non_standard(
            "MSSQL ignores trailing whitespace when comparing strings",
        )
        return x < y


with MSSqlTableImpl.op(ops.LessEqual()) as op:

    @op("str, str -> bool")
    def _le(x, y):
        warn_non_standard(
            "MSSQL ignores trailing whitespace when comparing strings",
        )
        return x <= y


with MSSqlTableImpl.op(ops.Greater()) as op:

    @op("str, str -> bool")
    def _gt(x, y):
        warn_non_standard(
            "MSSQL ignores trailing whitespace when comparing strings",
        )
        return x > y


with MSSqlTableImpl.op(ops.GreaterEqual()) as op:

    @op("str, str -> bool")
    def _ge(x, y):
        warn_non_standard(
            "MSSQL ignores trailing whitespace when comparing strings",
        )
        return x >= y


with MSSqlTableImpl.op(ops.Pow()) as op:

    @op.auto
    def _pow(lhs, rhs):
        # In MSSQL, the output type of pow is the same as the input type.
        # This means, that if lhs is a decimal, then we may very easily loose
        # a lot of precision if the exponent is <= 1
        # https://learn.microsoft.com/en-us/sql/t-sql/functions/power-transact-sql?view=sql-server-ver16
        return sa.func.POWER(sa.cast(lhs, sa.Double()), rhs, type_=sa.Double())


with MSSqlTableImpl.op(ops.RPow()) as op:

    @op.auto
    def _rpow(rhs, lhs):
        return _pow(lhs, rhs)


with MSSqlTableImpl.op(ops.StrLen()) as op:

    @op.auto
    def _str_length(x):
        warn_non_standard(
            "MSSQL ignores trailing whitespace when computing string length",
        )
        return sa.func.LENGTH(x, type_=sa.Integer())


with MSSqlTableImpl.op(ops.StrReplaceAll()) as op:

    @op.auto
    def _replace(x, y, z):
        x = x.collate("Latin1_General_CS_AS")
        return sa.func.REPLACE(x, y, z, type_=x.type)


with MSSqlTableImpl.op(ops.StrStartsWith()) as op:

    @op.auto
    def _startswith(x, y):
        x = x.collate("Latin1_General_CS_AS")
        return x.startswith(y, autoescape=True)


with MSSqlTableImpl.op(ops.StrEndsWith()) as op:

    @op.auto
    def _endswith(x, y):
        x = x.collate("Latin1_General_CS_AS")
        return x.endswith(y, autoescape=True)


with MSSqlTableImpl.op(ops.StrContains()) as op:

    @op.auto
    def _contains(x, y):
        x = x.collate("Latin1_General_CS_AS")
        return x.contains(y, autoescape=True)


with MSSqlTableImpl.op(ops.StrSlice()) as op:

    @op.auto
    def _str_slice(x, offset, length):
        return sa.func.SUBSTRING(x, offset + 1, length)


with MSSqlTableImpl.op(ops.DtDayOfWeek()) as op:

    @op.auto
    def _day_of_week(x):
        # Offset DOW such that Mon=1, Sun=7
        _1 = sa.literal_column("1")
        _2 = sa.literal_column("2")
        _7 = sa.literal_column("7")
        return (sa.extract("dow", x) + sa.text("@@DATEFIRST") - _2) % _7 + _1


with MSSqlTableImpl.op(ops.Mean()) as op:

    @op.auto
    def _mean(x):
        return sa.func.AVG(sa.cast(x, sa.Double()), type_=sa.Double())
