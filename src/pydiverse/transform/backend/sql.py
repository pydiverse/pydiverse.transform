from __future__ import annotations

import dataclasses
import functools
import inspect
import operator
from typing import Any

import polars as pl
import sqlalchemy as sqa

from pydiverse.transform import ops
from pydiverse.transform.backend.table_impl import TableImpl
from pydiverse.transform.backend.targets import Polars, SqlAlchemy, Target
from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree import dtypes, verbs
from pydiverse.transform.tree.col_expr import (
    Col,
    ColExpr,
    ColFn,
    ColName,
    LiteralCol,
    Order,
)
from pydiverse.transform.tree.dtypes import DType
from pydiverse.transform.tree.table_expr import TableExpr


class SqlImpl(TableImpl):
    Dialects: dict[str, type[TableImpl]] = {}

    def __new__(cls, *args, **kwargs) -> SqlImpl:
        engine: str | sqa.Engine = (
            inspect.signature(cls.__init__)
            .bind(None, *args, **kwargs)
            .arguments["conf"]
            .engine
        )

        dialect = (
            engine.dialect.name
            if isinstance(engine, sqa.Engine)
            else sqa.make_url(engine).get_dialect().name
        )

        return super().__new__(SqlImpl.Dialects[dialect])

    def __init__(self, table: str | sqa.Engine, conf: SqlAlchemy):
        assert type(self) is not SqlImpl
        self.engine = (
            conf.engine
            if isinstance(conf.engine, sqa.Engine)
            else sqa.create_engine(conf.engine)
        )
        self.table = sqa.Table(
            table, sqa.MetaData(), schema=conf.schema, autoload_with=self.engine
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        SqlImpl.Dialects[cls.dialect_name] = cls

    @staticmethod
    def export(expr: TableExpr, target: Target) -> Any:
        engine = get_engine(expr)
        table, query = compile_table_expr(expr)
        sel = compile_query(table, query)
        if isinstance(target, Polars):
            with engine.connect() as conn:
                return pl.read_database(sel, connection=conn)

        raise NotImplementedError

    @staticmethod
    def build_query(expr: TableExpr) -> str | None:
        engine = get_engine(expr)
        sel = compile_query(*compile_table_expr(expr))
        return str(
            sel.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True})
        )

    def col_names(self) -> list[str]:
        return [col.name for col in self.table.columns]

    def schema(self) -> dict[str, DType]:
        return {col.name: sqa_type_to_pdt(col.type) for col in self.table.columns}


# checks that all leafs use the same sqa.Engine and returns it
def get_engine(expr: TableExpr) -> sqa.Engine:
    if isinstance(expr, verbs.Join):
        engine = get_engine(expr.left)
        right_engine = get_engine(expr.right)
        if engine != right_engine:
            raise NotImplementedError  # TODO: find some good error for this
    elif isinstance(expr, Table):
        engine = expr._impl.engine
    else:
        engine = get_engine(expr.table)

    return engine


def compile_order(
    order: Order,
    name_to_sqa_col: dict[str, sqa.ColumnElement],
    group_by: list[sqa.ColumnElement],
) -> sqa.UnaryExpression:
    order_expr = compile_col_expr(order.order_by, name_to_sqa_col, group_by)
    order_expr = order_expr.desc() if order.descending else order_expr.asc()
    order_expr = (
        order_expr.nulls_last() if order.nulls_last else order_expr.nulls_first()
    )
    return order_expr


def compile_col_expr(
    expr: ColExpr,
    name_to_sqa_col: dict[str, sqa.ColumnElement],
    group_by: sqa.sql.expression.ClauseList,
) -> sqa.ColumnElement:
    assert not isinstance(expr, Col)
    if isinstance(expr, ColName):
        # here, inserted columns referenced via C are implicitly expanded
        return name_to_sqa_col[expr.name]
    elif isinstance(expr, ColFn):
        args: list[sqa.ColumnElement] = [
            compile_col_expr(arg, name_to_sqa_col, group_by) for arg in expr.args
        ]
        impl = SqlImpl.operator_registry.get_implementation(
            expr.name, tuple(arg.dtype for arg in expr.args)
        )

        partition_by = expr.context_kwargs.get("partition_by")
        if partition_by is not None:
            partition_by = sqa.sql.expression.ClauseList(
                *(compile_col_expr(col, name_to_sqa_col, []) for col in partition_by)
            )

        arrange = expr.context_kwargs.get("arrange")

        if arrange:
            order_by = sqa.sql.expression.ClauseList(
                *(compile_order(order, name_to_sqa_col, group_by) for order in arrange)
            )
        else:
            order_by = None

        filter_cond = expr.context_kwargs.get("filter")
        if filter_cond:
            filter_cond = [compile_col_expr(z, []) for z in filter_cond]
            raise NotImplementedError

        value: sqa.ColumnElement = impl(*args)

        if partition_by or order_by:
            value = value.over(partition_by=partition_by, order_by=order_by)

        return value

    elif isinstance(expr, LiteralCol):
        return sqa.literal(expr.val, type_=pdt_type_to_sqa(expr.dtype))

    raise AssertionError


# the compilation function only deals with one subquery. It assumes that any col
# it uses that is created by a subquery has the string name given to it in the
# name propagation stage. A subquery is thus responsible for inserting the right
# `AS` in the `SELECT` clause.


@dataclasses.dataclass(slots=True)
class Query:
    name_to_sqa_col: dict[str, sqa.ColumnElement]
    select: list[tuple[ColExpr, str]]
    join: list[SqlJoin] = dataclasses.field(default_factory=list)
    group_by: list[ColName] = dataclasses.field(default_factory=list)
    partition_by: list[ColName] = dataclasses.field(default_factory=list)
    where: list[ColExpr] = dataclasses.field(default_factory=list)
    having: list[ColExpr] = dataclasses.field(default_factory=list)
    order_by: list[Order] = dataclasses.field(default_factory=list)
    limit: int | None = None
    offset: int | None = None


@dataclasses.dataclass(slots=True)
class SqlJoin:
    right: sqa.Subquery
    on: ColExpr
    how: str


def compile_query(table: sqa.Table, query: Query) -> sqa.sql.Select:
    sel = table.select().select_from(table)

    for j in query.join:
        compiled_on = compile_col_expr(j.on, query.name_to_sqa_col, query.partition_by)
        sel = sel.join(
            j.right,
            onclause=compiled_on,
            isouter=j.how != "inner",
            full=j.how == "outer",
        )

    if query.where:
        where_cond = functools.reduce(operator.and_, query.where)
        sel = sel.where(
            compile_col_expr(where_cond, query.name_to_sqa_col, query.partition_by)
        )

    if query.group_by:
        sel = sel.group_by(
            *(
                compile_col_expr(col, query.name_to_sqa_col, query.partition_by)
                for col in query.group_by
            )
        )

    if query.having:
        having_cond = functools.reduce(operator.and_, query.having)
        sel = sel.having(
            compile_col_expr(having_cond, query.name_to_sqa_col, query.partition_by)
        )

    if query.limit is not None:
        sel = sel.limit(query.limit).offset(query.offset)

    sel = sel.with_only_columns(
        *(
            compile_col_expr(col_expr, query.name_to_sqa_col, query.partition_by).label(
                col_name
            )
            for col_expr, col_name in query.select
        )
    )

    if query.order_by:
        sel = sel.order_by(
            *(
                compile_order(ord, query.name_to_sqa_col, query.partition_by)
                for ord in query.order_by
            )
        )

    return sel


def compile_table_expr(expr: TableExpr) -> tuple[sqa.Table, Query]:
    if isinstance(expr, verbs.Select):
        table, query = compile_table_expr(expr.table)
        query.select = [(col, col.name) for col in expr.selected]

    if isinstance(expr, verbs.Drop):
        table, query = compile_table_expr(expr.table)
        query.select = [
            (col, name) for col, name in query.select if name not in set(expr.dropped)
        ]

    elif isinstance(expr, verbs.Rename):
        table, query = compile_table_expr(expr.table)
        query.name_to_sqa_col = {
            (expr.name_map[name] if name in expr.name_map else name): col
            for name, col in query.name_to_sqa_col.items()
        }

    elif isinstance(expr, verbs.Mutate):
        table, query = compile_table_expr(expr.table)
        query.select.extend([(val, name) for val, name in zip(expr.values, expr.names)])
        query.name_to_sqa_col.update(
            {
                name: compile_col_expr(val, query.name_to_sqa_col, query.partition_by)
                for name, val in zip(expr.names, expr.values)
            }
        )

    elif isinstance(expr, verbs.Join):
        table, query = compile_table_expr(expr.left)
        right_table, right_query = compile_table_expr(expr.right)

        j = SqlJoin(right_table, expr.on, expr.how)

        if expr.how == "inner":
            query.where.extend(right_query.where)
        elif expr.how == "left":
            j.on = functools.reduce(operator.and_, (j.on, *right_query.where))
        elif expr.how == "outer":
            if query.where or right_query.where:
                raise ValueError("invalid filter before outer join")

        query.select.extend(
            (ColName(name + expr.suffix), name + expr.suffix)
            for col, name in right_query.select
        )
        query.join.append(j)
        query.name_to_sqa_col.update(
            {
                name + expr.suffix: col_elem
                for name, col_elem in right_query.name_to_sqa_col.items()
            }
        )

    elif isinstance(expr, verbs.Filter):
        table, query = compile_table_expr(expr.table)

        if query.group_by:
            # check whether we can move conditions from `having` clause to `where`. This
            # is possible if a condition only involves columns in `group_by`. Split up
            # the filter at __and__`s until no longer possible. TODO
            query.having.extend(expr.filters)
        else:
            query.where.extend(expr.filters)

    elif isinstance(expr, verbs.Arrange):
        table, query = compile_table_expr(expr.table)
        # TODO: we could remove duplicates here if we want. but if we do so, this should
        # not be done in the sql backend but on the abstract tree.
        query.order_by = expr.order_by + query.order_by

    elif isinstance(expr, verbs.Summarise):
        table, query = compile_table_expr(expr.table)
        if query.group_by:
            assert query.group_by == query.partition_by
        query.group_by = query.partition_by
        query.partition_by = []
        query.select = [(col, col.name) for col in query.group_by] + [
            (val, name) for val, name in zip(expr.values, expr.names)
        ]
        query.name_to_sqa_col.update(
            {
                name: compile_col_expr(val, query.name_to_sqa_col, query.partition_by)
                for name, val in zip(expr.names, expr.values)
            }
        )

    elif isinstance(expr, verbs.SliceHead):
        table, query = compile_table_expr(expr.table)
        if query.limit is None:
            query.limit = expr.n
            query.offset = expr.offset
        else:
            query.limit = min(abs(query.limit - expr.offset), expr.n)
            query.offset += expr.offset

    elif isinstance(expr, verbs.GroupBy):
        table, query = compile_table_expr(expr.table)
        if expr.add:
            query.partition_by += expr.group_by
        else:
            query.partition_by = expr.group_by

    elif isinstance(expr, verbs.Ungroup):
        table, query = compile_table_expr(expr.table)
        assert not (query.partition_by and query.group_by)
        query.partition_by = []

    elif isinstance(expr, Table):
        return expr._impl.table, Query(
            {col.name: col for col in expr._impl.table.columns},
            [(ColName(col_name), col_name) for col_name in expr.col_names()],
        )

    return table, query


def sqa_type_to_pdt(t: sqa.types.TypeEngine) -> DType:
    if isinstance(t, sqa.Integer):
        return dtypes.Int()
    if isinstance(t, sqa.Numeric):
        return dtypes.Float()
    if isinstance(t, sqa.String):
        return dtypes.String()
    if isinstance(t, sqa.Boolean):
        return dtypes.Bool()
    if isinstance(t, sqa.DateTime):
        return dtypes.DateTime()
    if isinstance(t, sqa.Date):
        return dtypes.Date()
    if isinstance(t, sqa.Interval):
        return dtypes.Duration()

    raise TypeError(f"SQLAlchemy type {t} not supported by pydiverse.transform")


def pdt_type_to_sqa(t: DType) -> sqa.types.TypeEngine:
    if isinstance(t, dtypes.Int):
        return sqa.Integer()
    if isinstance(t, dtypes.Float):
        return sqa.Numeric()
    if isinstance(t, dtypes.String):
        return sqa.String()
    if isinstance(t, dtypes.Bool):
        return sqa.Boolean()
    if isinstance(t, dtypes.DateTime):
        return sqa.DateTime()
    if isinstance(t, dtypes.Date):
        return sqa.Date()
    if isinstance(t, dtypes.Duration):
        return sqa.Interval()

    raise AssertionError


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
