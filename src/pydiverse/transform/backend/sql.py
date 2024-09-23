from __future__ import annotations

import dataclasses
import functools
import inspect
import itertools
import operator
from collections.abc import Iterable
from typing import Any
from uuid import UUID

import polars as pl
import sqlalchemy as sqa

from pydiverse.transform import ops
from pydiverse.transform.backend.polars import pdt_type_to_polars
from pydiverse.transform.backend.table_impl import TableImpl
from pydiverse.transform.backend.targets import Polars, SqlAlchemy, Target
from pydiverse.transform.ops.core import Ftype
from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree import dtypes, verbs
from pydiverse.transform.tree.col_expr import (
    CaseExpr,
    Col,
    ColExpr,
    ColFn,
    LiteralCol,
    Order,
)
from pydiverse.transform.tree.dtypes import Dtype
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

    def col_names(self) -> list[str]:
        return [col.name for col in self.table.columns]

    def schema(self) -> dict[str, Dtype]:
        return {col.name: sqa_type_to_pdt(col.type) for col in self.table.columns}

    def clone(self) -> SqlImpl:
        cloned = object.__new__(self.__class__)
        cloned.engine = self.engine
        cloned.table = self.table
        return cloned

    @classmethod
    def build_select(cls, expr: TableExpr) -> sqa.Select:
        create_aliases(expr, {})
        table, query, sqa_col = cls.compile_table_expr(
            expr, {col.uuid: 1 for col in expr._select}
        )
        return cls.compile_query(
            table, query, (sqa_col[col.uuid] for col in expr._select)
        )

    @classmethod
    def export(cls, expr: TableExpr, target: Target) -> Any:
        sel = cls.build_select(expr)
        engine = get_engine(expr)
        if isinstance(target, Polars):
            with engine.connect() as conn:
                # TODO: Provide schema_overrides to not get u32 and other unwanted
                # integer / float types
                return pl.read_database(
                    sel,
                    connection=conn,
                    schema_overrides={
                        col.name: pdt_type_to_polars(col.dtype())
                        for col in expr._select
                    },
                )

        raise NotImplementedError

    @classmethod
    def build_query(cls, expr: TableExpr) -> str | None:
        sel = cls.build_select(expr)
        engine = get_engine(expr)
        return str(
            sel.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True})
        )

    @classmethod
    def compile_order(
        cls, order: Order, sqa_col: dict[str, sqa.Label]
    ) -> sqa.UnaryExpression:
        order_expr = cls.compile_col_expr(order.order_by, sqa_col)
        order_expr = order_expr.desc() if order.descending else order_expr.asc()
        if order.nulls_last is not None:
            order_expr = (
                order_expr.nulls_last()
                if order.nulls_last
                else order_expr.nulls_first()
            )
        return order_expr

    @classmethod
    def compile_col_expr(
        cls, expr: ColExpr, sqa_col: dict[str, sqa.Label]
    ) -> sqa.ColumnElement:
        if isinstance(expr, Col):
            return sqa_col[expr.uuid]

        elif isinstance(expr, ColFn):
            args: list[sqa.ColumnElement] = [
                cls.compile_col_expr(arg, sqa_col) for arg in expr.args
            ]
            impl = cls.registry.get_impl(
                expr.name, tuple(arg.dtype() for arg in expr.args)
            )

            partition_by = expr.context_kwargs.get("partition_by")
            if partition_by is not None:
                partition_by = sqa.sql.expression.ClauseList(
                    *(cls.compile_col_expr(col, sqa_col) for col in partition_by)
                )

            arrange = expr.context_kwargs.get("arrange")

            if arrange:
                order_by = sqa.sql.expression.ClauseList(
                    *(
                        dedup_order_by(
                            cls.compile_order(order, sqa_col) for order in arrange
                        )
                    )
                )
            else:
                order_by = None

            # we need this since some backends cannot do `any` / `all` as a window
            # function, so we need to emulate it via `max` / `min`.
            if (partition_by is not None or order_by is not None) and (
                window_impl := impl.get_variant("window")
            ):
                value = window_impl(*args, partition_by=partition_by, order_by=order_by)

            else:
                value: sqa.ColumnElement = impl(*args)
                if partition_by is not None or order_by is not None:
                    value = value.over(partition_by=partition_by, order_by=order_by)

            return value

        elif isinstance(expr, CaseExpr):
            return sqa.case(
                *(
                    (
                        cls.compile_col_expr(cond, sqa_col),
                        cls.compile_col_expr(val, sqa_col),
                    )
                    for cond, val in expr.cases
                ),
                else_=(
                    cls.compile_col_expr(expr.default_val, sqa_col)
                    if expr.default_val is not None
                    else None
                ),
            )

        elif isinstance(expr, LiteralCol):
            return expr.val

        raise AssertionError

    @classmethod
    def compile_query(
        cls, table: sqa.Table, query: Query, select: Iterable[sqa.Label]
    ) -> sqa.sql.Select:
        sel = table.select().select_from(table)

        for j in query.join:
            sel = sel.join(
                j.right,
                onclause=j.on,
                isouter=j.how != "inner",
                full=j.how == "outer",
            )

        if query.where:
            sel = sel.where(*query.where)

        if query.group_by:
            sel = sel.group_by(*query.group_by)

        if query.having:
            sel = sel.having(*query.having)

        if query.limit is not None:
            sel = sel.limit(query.limit).offset(query.offset)

        if query.order_by:
            sel = sel.order_by(*query.order_by)

        sel = sel.with_only_columns(*select)

        return sel

    @classmethod
    def compile_table_expr(
        cls, expr: TableExpr, needed_cols: dict[UUID, int]
    ) -> tuple[sqa.Table, Query, dict[UUID, sqa.Label]]:
        if isinstance(expr, verbs.Verb):
            # store a counter how often each UUID is referenced by ancestors. This
            # allows to only select necessary columns in a subquery.
            for node in expr._iter_col_nodes():
                if isinstance(node, Col):
                    cnt = needed_cols.get(node.uuid)
                    if cnt is None:
                        needed_cols[node.uuid] = 1
                    else:
                        needed_cols[node.uuid] = cnt + 1

            table, query, sqa_col = cls.compile_table_expr(expr.table, needed_cols)

        # check if a subquery is required
        if (
            (
                isinstance(
                    expr,
                    (
                        verbs.Filter,
                        verbs.Summarise,
                        verbs.Arrange,
                        verbs.GroupBy,
                        verbs.Join,
                    ),
                )
                and query.limit is not None
            )
            or (
                isinstance(expr, (verbs.Mutate, verbs.Filter))
                and any(
                    node.ftype(agg_is_window=True) == Ftype.WINDOW
                    for node in expr._iter_col_nodes()
                    if isinstance(node, Col)
                )
            )
            or (
                isinstance(expr, verbs.Summarise)
                and (
                    (bool(query.group_by) and set(query.group_by) != query.partition_by)
                    or any(
                        (
                            node.ftype(agg_is_window=False)
                            in (Ftype.WINDOW, Ftype.AGGREGATE)
                        )
                        for node in expr._iter_col_nodes()
                        if isinstance(node, Col)
                    )
                )
            )
        ):
            if needed_cols.keys().isdisjoint(sqa_col.keys()):
                # We cannot select zero columns from a subquery. This happens when the
                # user only 0-ary functions after the subquery, e.g. `count`.
                needed_cols[next(iter(sqa_col.keys()))] = 1

            # TODO: do we want `alias` to automatically create a subquery? or add a
            # flag to the node that a subquery would be allowed? or special verb to
            # mark subquery?

            # We only want to select those columns that (1) the user uses in some
            # expression later or (2) are present in the final selection.
            table = cls.compile_query(
                table,
                query,
                (sqa_col[uid] for uid in needed_cols.keys() if uid in sqa_col),
            ).subquery()
            sqa_col.update(
                {
                    uid: sqa.label(
                        sqa_col[uid].name, table.columns.get(sqa_col[uid].name)
                    )
                    for uid in needed_cols.keys()
                    if uid in sqa_col
                }
            )
            # rewire col refs to the subquery
            query = Query(
                partition_by=[table.columns.get(col.name) for col in query.partition_by]
            )

        if isinstance(expr, verbs.Rename):
            sqa_col = {
                uid: (
                    sqa.label(expr.name_map[lb.name], lb)
                    if lb.name in expr.name_map
                    else lb
                )
                for uid, lb in sqa_col.items()
            }

        elif isinstance(expr, verbs.Mutate):
            for name, val in zip(expr.names, expr.values):
                sqa_col[expr._name_to_uuid[name]] = sqa.label(
                    name, cls.compile_col_expr(val, sqa_col)
                )

        elif isinstance(expr, verbs.Filter):
            if query.group_by:
                query.having.extend(
                    cls.compile_col_expr(fil, sqa_col) for fil in expr.filters
                )
            else:
                query.where.extend(
                    cls.compile_col_expr(fil, sqa_col) for fil in expr.filters
                )

        elif isinstance(expr, verbs.Arrange):
            query.order_by = dedup_order_by(
                itertools.chain(
                    (cls.compile_order(ord, sqa_col) for ord in expr.order_by),
                    query.order_by,
                )
            )

        elif isinstance(expr, verbs.Summarise):
            query.group_by.extend(query.partition_by)

            for name, val in zip(expr.names, expr.values):
                sqa_col[expr._name_to_uuid[name]] = sqa.Label(
                    name, cls.compile_col_expr(val, sqa_col)
                )

            query.partition_by = []
            query.order_by.clear()

        elif isinstance(expr, verbs.SliceHead):
            if query.limit is None:
                query.limit = expr.n
                query.offset = expr.offset
            else:
                query.limit = min(abs(query.limit - expr.offset), expr.n)
                query.offset += expr.offset

        elif isinstance(expr, verbs.GroupBy):
            compiled_group_by = (
                sqa.label(
                    col.name,
                    cls.compile_col_expr(col, sqa_col),
                )
                for col in expr.group_by
            )
            if expr.add:
                query.partition_by.extend(compiled_group_by)
            else:
                query.partition_by = list(compiled_group_by)

        elif isinstance(expr, verbs.Ungroup):
            assert not (query.partition_by and query.group_by)
            query.partition_by.clear()

        elif isinstance(expr, verbs.Join):
            right_table, right_query, right_sqa_col = cls.compile_table_expr(
                expr.right, needed_cols
            )

            sqa_col.update(
                {
                    uid: sqa.label(lb.name + expr.suffix, lb)
                    for uid, lb in right_sqa_col.items()
                }
            )

            j = SqlJoin(
                right_table,
                cls.compile_col_expr(expr.on, sqa_col),
                expr.how,
            )

            if expr.how == "inner":
                query.where.extend(right_query.where)
            elif expr.how == "left":
                j.on = functools.reduce(operator.and_, (j.on, *right_query.where))
            elif expr.how == "outer":
                if query.where or right_query.where:
                    raise ValueError("invalid filter before outer join")

            query.join.append(j)

        elif isinstance(expr, Table):
            table = expr._impl.table
            query = Query()
            sqa_col = {
                expr._name_to_uuid[col.name]: sqa.label(col.name, col)
                for col in expr._impl.table.columns
            }

        if isinstance(expr, verbs.Verb):
            # decrease counters (`needed_cols` is not copied)
            for node in expr._iter_col_nodes():
                if isinstance(node, Col):
                    cnt = needed_cols.get(node.uuid)
                    if cnt == 1:
                        del needed_cols[node.uuid]
                    else:
                        needed_cols[node.uuid] = cnt - 1

        return table, query, sqa_col


@dataclasses.dataclass(slots=True)
class Query:
    join: list[SqlJoin] = dataclasses.field(default_factory=list)
    group_by: list[sqa.Label] = dataclasses.field(default_factory=list)
    partition_by: list[sqa.Label] = dataclasses.field(default_factory=list)
    where: list[sqa.ColumnElement] = dataclasses.field(default_factory=list)
    having: list[sqa.ColumnElement] = dataclasses.field(default_factory=list)
    order_by: list[sqa.UnaryExpression] = dataclasses.field(default_factory=list)
    limit: int | None = None
    offset: int | None = None


@dataclasses.dataclass(slots=True)
class SqlJoin:
    right: sqa.Subquery
    on: sqa.ColumnElement
    how: verbs.JoinHow


# MSSQL complains about duplicates in ORDER BY.
def dedup_order_by(
    order_by: Iterable[sqa.UnaryExpression],
) -> list[sqa.UnaryExpression]:
    new_order_by: list[sqa.UnaryExpression] = []
    occurred: set[sqa.ColumnElement] = set()

    for ord in order_by:
        peeled = ord
        while isinstance(peeled, sqa.UnaryExpression) and peeled.modifier is not None:
            peeled = peeled.element
        if peeled not in occurred:
            new_order_by.append(ord)
            occurred.add(peeled)

    return new_order_by


# Gives any leaf a unique alias to allow self-joins. We do this here to not force
# the user to come up with dummy names that are not required later anymore. It has
# to be done before a join so that all column references in the join subtrees remain
# valid.
def create_aliases(expr: TableExpr, num_occurences: dict[str, int]) -> dict[str, int]:
    if isinstance(expr, verbs.Verb):
        num_occurences = create_aliases(expr.table, num_occurences)

        if isinstance(expr, verbs.Join):
            num_occurences = create_aliases(expr.right, num_occurences)

    elif isinstance(expr, Table):
        if cnt := num_occurences.get(expr._impl.table.name):
            expr._impl.table = expr._impl.table.alias(f"{expr._impl.table.name}_{cnt}")
        else:
            cnt = 0
        num_occurences[expr._impl.table.name] = cnt + 1

    else:
        raise AssertionError

    return num_occurences


def get_engine(expr: TableExpr) -> sqa.Engine:
    if isinstance(expr, verbs.Verb):
        engine = get_engine(expr.table)

        if isinstance(expr, verbs.Join):
            right_engine = get_engine(expr.right)
            if engine != right_engine:
                raise NotImplementedError  # TODO: find some good error for this

    else:
        assert isinstance(expr, Table)
        engine = expr._impl.engine

    return engine


def sqa_type_to_pdt(t: sqa.types.TypeEngine) -> Dtype:
    if isinstance(t, sqa.Integer):
        return dtypes.Int()
    elif isinstance(t, sqa.Numeric):
        return dtypes.Float()
    elif isinstance(t, sqa.String):
        return dtypes.String()
    elif isinstance(t, sqa.Boolean):
        return dtypes.Bool()
    elif isinstance(t, sqa.DateTime):
        return dtypes.DateTime()
    elif isinstance(t, sqa.Date):
        return dtypes.Date()
    elif isinstance(t, sqa.Interval):
        return dtypes.Duration()
    elif isinstance(t, sqa.Null):
        return dtypes.NoneDtype()

    raise TypeError(f"SQLAlchemy type {t} not supported by pydiverse.transform")


def pdt_type_to_sqa(t: Dtype) -> sqa.types.TypeEngine:
    if isinstance(t, dtypes.Int):
        return sqa.Integer()
    elif isinstance(t, dtypes.Float):
        return sqa.Numeric()
    elif isinstance(t, dtypes.String):
        return sqa.String()
    elif isinstance(t, dtypes.Bool):
        return sqa.Boolean()
    elif isinstance(t, dtypes.DateTime):
        return sqa.DateTime()
    elif isinstance(t, dtypes.Date):
        return sqa.Date()
    elif isinstance(t, dtypes.Duration):
        return sqa.Interval()
    elif isinstance(t, dtypes.NoneDtype):
        return sqa.types.NullType()

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
    def _isin(x, *values):
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
    def _replace_all(x, y, z):
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
        return sqa.func.coalesce(sqa.func.max(x), sqa.null())

    @op.auto(variant="window")
    def _any(x, *, partition_by=None, order_by=None):
        return sqa.func.coalesce(
            sqa.func.max(x).over(
                partition_by=partition_by,
                order_by=order_by,
            ),
            sqa.null(),
        )


with SqlImpl.op(ops.All()) as op:

    @op.auto
    def _all(x):
        return sqa.func.coalesce(sqa.func.min(x), sqa.null())

    @op.auto(variant="window")
    def _all(x, *, partition_by=None, order_by=None):
        return sqa.func.coalesce(
            sqa.func.min(x).over(
                partition_by=partition_by,
                order_by=order_by,
            ),
            sqa.null(),
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
        partition_by=None,
        order_by=None,
    ):
        if by == 0:
            return x
        if by > 0:
            return sqa.func.LAG(x, by, empty_value, type_=x.type).over(
                partition_by=partition_by, order_by=order_by
            )
        if by < 0:
            return sqa.func.LEAD(x, -by, empty_value, type_=x.type).over(
                partition_by=partition_by, order_by=order_by
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
