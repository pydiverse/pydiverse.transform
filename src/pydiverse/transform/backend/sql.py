from __future__ import annotations

import functools
import operator
from typing import Any

import sqlalchemy as sqa

from pydiverse.transform import ops
from pydiverse.transform.backend.table_impl import TableImpl
from pydiverse.transform.backend.targets import Target
from pydiverse.transform.tree import verbs
from pydiverse.transform.tree.col_expr import ColExpr
from pydiverse.transform.tree.table_expr import TableExpr  #


class SqlImpl(TableImpl):
    @staticmethod
    def export(expr: TableExpr, target: Target) -> Any: ...


# the compilation function only deals with one subquery. It assumes that any col
# it uses that is created by a subquery has the string name given to it in the
# name propagation stage. A subquery is thus responsible for inserting the right
# `AS` in the `SELECT` clause.


class CompilationContext:
    select: list[tuple[ColExpr, str]]
    group_by: list[ColExpr]
    where: list[ColExpr]
    having: list[ColExpr]
    order_by: list[ColExpr]
    limit: int
    offset: int


def compile_subquery(expr: TableExpr) -> CompilationContext:
    if isinstance(expr, verbs.Select):
        ct = compile_subquery(expr.table)
        ct.select = [(col, col.name) for col in expr.selects]

    elif isinstance(expr, verbs.Rename):
        ...

    elif isinstance(expr, verbs.Mutate):
        ct = compile_subquery(expr.table)
        ct.select.extend([(val, name) for val, name in zip(expr.values, expr.names)])

    return ct


with SqlImpl.op(ops.FloorDiv(), check_super=False) as op:
    if sqa.__version__ < "2":

        @op.auto
        def _floordiv(lhs, rhs):
            return sqa.cast(lhs / rhs, sqa.Integer())

    else:

        @op.auto
        def _floordiv(lhs, rhs):
            return lhs // rhs


with SqlImpl.op(ops.RFloorDiv(), check_super=False) as op:

    @op.auto
    def _rfloordiv(rhs, lhs):
        return _floordiv(lhs, rhs)


with SqlImpl.op(ops.Pow()) as op:

    @op.auto
    def _pow(lhs, rhs):
        if isinstance(lhs.type, sqa.Float) or isinstance(rhs.type, sqa.Float):
            type_ = sqa.Double()
        elif isinstance(lhs.type, sqa.Numeric) or isinstance(rhs, sqa.Numeric):
            type_ = sqa.Numeric()
        else:
            type_ = sqa.Double()

        return sqa.func.POW(lhs, rhs, type_=type_)


with SqlImpl.op(ops.RPow()) as op:

    @op.auto
    def _rpow(rhs, lhs):
        return _pow(lhs, rhs)


with SqlImpl.op(ops.Xor()) as op:

    @op.auto
    def _xor(lhs, rhs):
        return lhs != rhs


with SqlImpl.op(ops.RXor()) as op:

    @op.auto
    def _rxor(rhs, lhs):
        return lhs != rhs


with SqlImpl.op(ops.Pos()) as op:

    @op.auto
    def _pos(x):
        return x


with SqlImpl.op(ops.Abs()) as op:

    @op.auto
    def _abs(x):
        return sqa.func.ABS(x, type_=x.type)


with SqlImpl.op(ops.Round()) as op:

    @op.auto
    def _round(x, decimals=0):
        return sqa.func.ROUND(x, decimals, type_=x.type)


with SqlImpl.op(ops.IsIn()) as op:

    @op.auto
    def _isin(x, *values, _verb=None):
        if _verb == "filter":
            # In WHERE and HAVING clause, we can use the IN operator
            return x.in_(values)
        # In SELECT we must replace it with the corresponding boolean expression
        return functools.reduce(operator.or_, map(lambda v: x == v, values))


with SqlImpl.op(ops.IsNull()) as op:

    @op.auto
    def _is_null(x):
        return x.is_(sqa.null())


with SqlImpl.op(ops.IsNotNull()) as op:

    @op.auto
    def _is_not_null(x):
        return x.is_not(sqa.null())


#### String Functions ####


with SqlImpl.op(ops.StrStrip()) as op:

    @op.auto
    def _str_strip(x):
        return sqa.func.TRIM(x, type_=x.type)


with SqlImpl.op(ops.StrLen()) as op:

    @op.auto
    def _str_length(x):
        return sqa.func.LENGTH(x, type_=sqa.Integer())


with SqlImpl.op(ops.StrToUpper()) as op:

    @op.auto
    def _upper(x):
        return sqa.func.UPPER(x, type_=x.type)


with SqlImpl.op(ops.StrToLower()) as op:

    @op.auto
    def _upper(x):
        return sqa.func.LOWER(x, type_=x.type)


with SqlImpl.op(ops.StrReplaceAll()) as op:

    @op.auto
    def _replace(x, y, z):
        return sqa.func.REPLACE(x, y, z, type_=x.type)


with SqlImpl.op(ops.StrStartsWith()) as op:

    @op.auto
    def _startswith(x, y):
        return x.startswith(y, autoescape=True)


with SqlImpl.op(ops.StrEndsWith()) as op:

    @op.auto
    def _endswith(x, y):
        return x.endswith(y, autoescape=True)


with SqlImpl.op(ops.StrContains()) as op:

    @op.auto
    def _contains(x, y):
        return x.contains(y, autoescape=True)


with SqlImpl.op(ops.StrSlice()) as op:

    @op.auto
    def _str_slice(x, offset, length):
        # SQL has 1-indexed strings but we do it 0-indexed
        return sqa.func.SUBSTR(x, offset + 1, length)


#### Datetime Functions ####


with SqlImpl.op(ops.DtYear()) as op:

    @op.auto
    def _year(x):
        return sqa.extract("year", x)


with SqlImpl.op(ops.DtMonth()) as op:

    @op.auto
    def _month(x):
        return sqa.extract("month", x)


with SqlImpl.op(ops.DtDay()) as op:

    @op.auto
    def _day(x):
        return sqa.extract("day", x)


with SqlImpl.op(ops.DtHour()) as op:

    @op.auto
    def _hour(x):
        return sqa.extract("hour", x)


with SqlImpl.op(ops.DtMinute()) as op:

    @op.auto
    def _minute(x):
        return sqa.extract("minute", x)


with SqlImpl.op(ops.DtSecond()) as op:

    @op.auto
    def _second(x):
        return sqa.extract("second", x)


with SqlImpl.op(ops.DtMillisecond()) as op:

    @op.auto
    def _millisecond(x):
        return sqa.extract("milliseconds", x) % 1000


with SqlImpl.op(ops.DtDayOfWeek()) as op:

    @op.auto
    def _day_of_week(x):
        return sqa.extract("dow", x)


with SqlImpl.op(ops.DtDayOfYear()) as op:

    @op.auto
    def _day_of_year(x):
        return sqa.extract("doy", x)


#### Generic Functions ####


with SqlImpl.op(ops.Greatest()) as op:

    @op.auto
    def _greatest(*x):
        # TODO: Determine return type
        return sqa.func.GREATEST(*x)


with SqlImpl.op(ops.Least()) as op:

    @op.auto
    def _least(*x):
        # TODO: Determine return type
        return sqa.func.LEAST(*x)


#### Summarising Functions ####


with SqlImpl.op(ops.Mean()) as op:

    @op.auto
    def _mean(x):
        type_ = sqa.Numeric()
        if isinstance(x.type, sqa.Float):
            type_ = sqa.Double()

        return sqa.func.AVG(x, type_=type_)


with SqlImpl.op(ops.Min()) as op:

    @op.auto
    def _min(x):
        return sqa.func.min(x)


with SqlImpl.op(ops.Max()) as op:

    @op.auto
    def _max(x):
        return sqa.func.max(x)


with SqlImpl.op(ops.Sum()) as op:

    @op.auto
    def _sum(x):
        return sqa.func.sum(x)


with SqlImpl.op(ops.Any()) as op:

    @op.auto
    def _any(x, *, _window_partition_by=None, _window_order_by=None):
        return sqa.func.coalesce(sqa.func.max(x), sqa.false())

    @op.auto(variant="window")
    def _any(x, *, _window_partition_by=None, _window_order_by=None):
        return sqa.func.coalesce(
            sqa.func.max(x).over(
                partition_by=_window_partition_by,
                order_by=_window_order_by,
            ),
            sqa.false(),
        )


with SqlImpl.op(ops.All()) as op:

    @op.auto
    def _all(x):
        return sqa.func.coalesce(sqa.func.min(x), sqa.false())

    @op.auto(variant="window")
    def _all(x, *, _window_partition_by=None, _window_order_by=None):
        return sqa.func.coalesce(
            sqa.func.min(x).over(
                partition_by=_window_partition_by,
                order_by=_window_order_by,
            ),
            sqa.false(),
        )


with SqlImpl.op(ops.Count()) as op:

    @op.auto
    def _count(x=None):
        if x is None:
            # Get the number of rows
            return sqa.func.count()
        else:
            # Count non null values
            return sqa.func.count(x)


#### Window Functions ####


with SqlImpl.op(ops.Shift()) as op:

    @op.auto
    def _shift():
        raise RuntimeError("This is a stub")

    @op.auto(variant="window")
    def _shift(
        x,
        by,
        empty_value=None,
        *,
        _window_partition_by=None,
        _window_order_by=None,
    ):
        if by == 0:
            return x
        if by > 0:
            return sqa.func.LAG(x, by, empty_value, type_=x.type).over(
                partition_by=_window_partition_by, order_by=_window_order_by
            )
        if by < 0:
            return sqa.func.LEAD(x, -by, empty_value, type_=x.type).over(
                partition_by=_window_partition_by, order_by=_window_order_by
            )


with SqlImpl.op(ops.RowNumber()) as op:

    @op.auto
    def _row_number():
        return sqa.func.ROW_NUMBER(type_=sqa.Integer())


with SqlImpl.op(ops.Rank()) as op:

    @op.auto
    def _rank():
        return sqa.func.rank()


with SqlImpl.op(ops.DenseRank()) as op:

    @op.auto
    def _dense_rank():
        return sqa.func.dense_rank()
