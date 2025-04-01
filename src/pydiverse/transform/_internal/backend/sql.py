from __future__ import annotations

import dataclasses
import functools
import inspect
import itertools
import math
import operator
from collections.abc import Iterable
from typing import Any
from uuid import UUID

import polars as pl
import sqlalchemy as sqa

from pydiverse.common import String
from pydiverse.transform._internal.backend.table_impl import TableImpl
from pydiverse.transform._internal.backend.targets import Polars, SqlAlchemy, Target
from pydiverse.transform._internal.errors import SubqueryError
from pydiverse.transform._internal.ops import ops
from pydiverse.transform._internal.ops.op import Ftype
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
from pydiverse.transform._internal.tree.types import (
    Dtype,
    Float,
    Float64,
    Int,
    Int64,
    NullType,
)


class SqlImpl(TableImpl):
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

        # We don't want to import any SQL impls we don't use, so the mapping
        # name -> impl class is defined here.

        if dialect == "sqlite":
            from .sqlite import SqliteImpl

            Impl = SqliteImpl
        elif dialect == "mssql":
            from .mssql import MsSqlImpl

            Impl = MsSqlImpl

        elif dialect == "postgresql":
            from .postgres import PostgresImpl

            Impl = PostgresImpl

        elif dialect == "duckdb":
            from .duckdb import DuckDbImpl

            Impl = DuckDbImpl

        return super().__new__(Impl)

    def __init__(self, table: str | sqa.Table, conf: SqlAlchemy, name: str | None):
        assert type(self) is not SqlImpl

        self.engine = (
            conf.engine
            if isinstance(conf.engine, sqa.Engine)
            else sqa.create_engine(conf.engine)
        )
        if isinstance(table, str):
            self.table = sqa.Table(
                table, sqa.MetaData(), schema=conf.schema, autoload_with=self.engine
            )
        else:
            self.table = table

        if name is None:
            name = self.table.name

        super().__init__(
            name,
            {col.name: self.pdt_type(col.type) for col in self.table.columns},
        )

    def col_names(self) -> list[str]:
        return [col.name for col in self.table.columns]

    def schema(self) -> dict[str, Dtype]:
        return {col.name: self.pdt_type(col.type) for col in self.table.columns}

    def _clone(self) -> tuple[SqlImpl, dict[AstNode, AstNode], dict[UUID, UUID]]:
        cloned = self.__class__(self.table, SqlAlchemy(self.engine), self.name)
        return (
            cloned,
            {self: cloned},
            {
                self.cols[name]._uuid: cloned.cols[name]._uuid
                for name in self.cols.keys()
            },
        )

    @classmethod
    def inf(cls):
        return sqa.cast(sqa.literal("inf"), sqa.Double)

    @classmethod
    def nan(cls):
        return sqa.cast(sqa.literal("nan"), sqa.Double)

    @classmethod
    def default_collation(cls):
        return "POSIX"

    @classmethod
    def build_select(cls, nd: AstNode, final_select: list[Col]) -> sqa.Select:
        create_aliases(nd, {})
        nd, query, _ = cls.compile_ast(nd, {col._uuid: 1 for col in final_select})
        return cls.compile_query(nd, query)

    @classmethod
    def export(
        cls,
        nd: AstNode,
        target: Target,
        final_select: list[Col],
        schema_overrides: dict[Col, Any],
    ) -> Any:
        sel = cls.build_select(nd, final_select)
        engine = get_engine(nd)
        if isinstance(target, Polars):
            with engine.connect() as conn:
                df = pl.read_database(
                    sel,
                    connection=conn,
                    schema_overrides={
                        sql_col.name: schema_overrides[col]
                        if col in schema_overrides
                        else col.dtype().to_polars()
                        for sql_col, col in zip(
                            sel.columns.values(), final_select, strict=True
                        )
                        if col in schema_overrides
                        or types.without_const(col.dtype()) == NullType()
                    },
                )
                df.name = nd.name
                return df

        raise NotImplementedError

    @classmethod
    def build_query(
        cls, nd: AstNode, final_select: list[Col], dialect=None
    ) -> str | None:
        sel = cls.build_select(nd, final_select)
        if dialect is None:
            dialect = get_engine(nd).dialect
        return str(sel.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))

    # some backends need to do casting to ensure the correct type
    @classmethod
    def compile_lit(cls, lit: LiteralCol):
        if types.without_const(lit.dtype()).is_float():
            if math.isnan(lit.val):
                return cls.nan()
            elif math.isinf(lit.val):
                return cls.inf() if lit.val > 0 else -cls.inf()
        return sqa.literal(lit.val, cls.sqa_type(lit.dtype()))

    @classmethod
    def compile_order(
        cls, order: Order, sqa_col: dict[str, sqa.Label]
    ) -> sqa.UnaryExpression:
        order_expr = cls.compile_col_expr(order.order_by, sqa_col)
        if types.without_const(order.order_by.dtype()) == String():
            order_expr = order_expr.collate(cls.default_collation())
        order_expr = order_expr.desc() if order.descending else order_expr.asc()
        if order.nulls_last is not None:
            order_expr = (
                order_expr.nulls_last()
                if order.nulls_last
                else order_expr.nulls_first()
            )
        return order_expr

    @classmethod
    def compile_cast(cls, cast: Cast, sqa_col: dict[str, sqa.Label]) -> sqa.Cast:
        return cls.compile_col_expr(cast.val, sqa_col).cast(
            cls.sqa_type(cast.target_type)
        )

    @classmethod
    def fix_fn_types(
        cls, fn: ColFn, val: sqa.ColumnElement, *args: sqa.ColumnElement
    ) -> sqa.ColumnElement:
        return val

    @classmethod
    def compile_ordered_aggregation(
        cls, *args: sqa.ColumnElement, order_by: sqa.UnaryExpression, impl
    ) -> sqa.ColumnElement:
        raise NotImplementedError

    @classmethod
    def compile_col_expr(
        cls, expr: ColExpr, sqa_col: dict[str, sqa.Label], *, compile_literals=True
    ) -> sqa.ColumnElement:
        if isinstance(expr, Col):
            return sqa_col[expr._uuid]

        elif isinstance(expr, ColFn):
            args: list[sqa.ColumnElement] = [
                cls.compile_col_expr(
                    arg, sqa_col, compile_literals=not types.is_const(param)
                )
                for arg, param in zip(
                    expr.args,
                    expr.op.trie.best_match(tuple(arg.dtype() for arg in expr.args))[0],
                    strict=False,
                )
            ]

            assert expr.ftype() is not None

            partition_by = expr.context_kwargs.get("partition_by")
            if partition_by:
                assert expr.ftype() == Ftype.WINDOW
                partition_by = sqa.sql.expression.ClauseList(
                    *(cls.compile_col_expr(col, sqa_col) for col in partition_by)
                )

            arrange = expr.context_kwargs.get("arrange")
            if arrange:
                order_by = dedup_order_by(
                    cls.compile_order(order, sqa_col) for order in arrange
                )
            else:
                order_by = None

            impl = cls.get_impl(expr.op, tuple(arg.dtype() for arg in expr.args))
            impl = functools.partial(impl, _Impl=cls)

            if order_by is not None and expr.ftype() == Ftype.AGGREGATE:
                # some backends need to do preprocessing and some postprocessing here,
                # so we just give them full control by passing the responsibility of
                # calling the `impl`.
                value = cls.compile_ordered_aggregation(
                    *args, order_by=order_by, impl=impl
                )

            else:
                value: sqa.FunctionElement = impl(*args)

                if (
                    partition_by is not None
                    or order_by is not None
                    and expr.ftype() == Ftype.WINDOW
                ):
                    value = sqa.over(
                        value,
                        partition_by=partition_by,
                        order_by=sqa.sql.expression.ClauseList(*order_by)
                        if order_by
                        else None,
                    )

            return cls.fix_fn_types(expr, value, *args)

        elif isinstance(expr, CaseExpr):
            res = sqa.case(
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

            if not cls.pdt_type(res.type).is_subtype(expr.dtype()):
                res = res.cast(
                    cls.sqa_type(
                        Int64()
                        if type(expr.dtype()) is Int
                        else Float64()
                        if type(expr.dtype()) is Float
                        else expr.dtype()
                    )
                )

            return res

        elif isinstance(expr, LiteralCol):
            return cls.compile_lit(expr) if compile_literals else expr.val

        elif isinstance(expr, Cast):
            return cls.compile_cast(expr, sqa_col)

        raise AssertionError

    @classmethod
    def compile_query(cls, table: sqa.Table, query: Query) -> sqa.sql.Select:
        sel = table.select().select_from(table)

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

        sel = sel.with_only_columns(*query.select)

        return sel

    @classmethod
    def compile_ast(
        cls, nd: AstNode, needed_cols: dict[UUID, int]
    ) -> tuple[sqa.Table, Query, dict[UUID, sqa.Label]]:
        if isinstance(nd, verbs.Verb):
            # store a counter in `needed_cols how often each UUID is referenced by
            # ancestors. This allows to only select necessary columns in a subquery.
            for node in nd.iter_col_nodes():
                if isinstance(node, Col):
                    cnt = needed_cols.get(node._uuid)
                    if cnt is None:
                        needed_cols[node._uuid] = 1
                    else:
                        needed_cols[node._uuid] = cnt + 1

            table, query, sqa_col = cls.compile_ast(nd.child, needed_cols)
            table, query, sqa_col = cls.check_for_subquery(
                nd, table, sqa_col, query, needed_cols
            )

        if isinstance(nd, verbs.Mutate | verbs.Summarize):
            query.select = [lb for lb in query.select if lb.name not in set(nd.names)]

        if isinstance(nd, verbs.Select):
            query.select = [sqa_col[col._uuid] for col in nd.select]

        elif isinstance(nd, verbs.Rename):
            sqa_col = {
                uid: (
                    sqa.label(nd.name_map[lb.name], lb)
                    if lb.name in nd.name_map
                    else lb
                )
                for uid, lb in sqa_col.items()
            }

            query.select, query.partition_by, query.group_by = (
                [
                    sqa.label(nd.name_map[lb.name], lb)
                    if lb.name in nd.name_map
                    else lb
                    for lb in label_arr
                ]
                for label_arr in (query.select, query.partition_by, query.group_by)
            )

        elif isinstance(nd, verbs.Mutate):
            for name, val, uid in zip(nd.names, nd.values, nd.uuids, strict=True):
                sqa_col[uid] = sqa.label(name, cls.compile_col_expr(val, sqa_col))
                query.select.append(sqa_col[uid])

        elif isinstance(nd, verbs.Filter):
            if query.group_by:
                query.having.extend(
                    cls.compile_col_expr(fil, sqa_col) for fil in nd.predicates
                )
            else:
                query.where.extend(
                    cls.compile_col_expr(fil, sqa_col) for fil in nd.predicates
                )

        elif isinstance(nd, verbs.Arrange):
            query.order_by = dedup_order_by(
                itertools.chain(
                    (cls.compile_order(ord, sqa_col) for ord in nd.order_by),
                    query.order_by,
                )
            )

        elif isinstance(nd, verbs.Summarize):
            query.group_by.extend(query.partition_by)

            for name, val, uid in zip(nd.names, nd.values, nd.uuids, strict=True):
                sqa_col[uid] = sqa.Label(name, cls.compile_col_expr(val, sqa_col))

            query.select = query.partition_by + [sqa_col[uid] for uid in nd.uuids]
            query.partition_by = []
            query.order_by.clear()

        elif isinstance(nd, verbs.SliceHead):
            if query.limit is None:
                query.limit = nd.n
                query.offset = nd.offset
            else:
                query.limit = min(abs(query.limit - nd.offset), nd.n)
                query.offset += nd.offset

        elif isinstance(nd, verbs.GroupBy):
            compiled_group_by = (sqa_col[col._uuid] for col in nd.group_by)
            if nd.add:
                query.partition_by.extend(compiled_group_by)
            else:
                query.partition_by = list(compiled_group_by)

        elif isinstance(nd, verbs.Ungroup):
            assert not (query.partition_by and query.group_by)
            query.partition_by.clear()

        elif isinstance(nd, verbs.Join):
            right_table, right_query, right_sqa_col = cls.compile_ast(
                nd.right, needed_cols
            )
            right_table, right_query, right_sqa_col = cls.check_for_subquery(
                nd, right_table, right_sqa_col, right_query, needed_cols, child=nd.right
            )
            sqa_col.update(
                {
                    uid: sqa.label(lb.name + nd.suffix, lb)
                    for uid, lb in right_sqa_col.items()
                }
            )

            compiled_on = cls.compile_col_expr(nd.on, sqa_col)

            if nd.how == "inner":
                query.where.extend(right_query.where)
            elif nd.how == "left":
                compiled_on = functools.reduce(
                    operator.and_, (compiled_on, *right_query.where)
                )
            elif nd.how == "full":
                if query.where or right_query.where:
                    raise ValueError("invalid filter before full join")

            table = table.join(
                right_table,
                onclause=compiled_on,
                isouter=nd.how != "inner",
                full=nd.how == "full",
            )

            query.select += [
                sqa.Label(lb.name + nd.suffix, lb) for lb in right_query.select
            ]

            assert not right_query.partition_by
            assert not right_query.group_by

        elif isinstance(nd, TableImpl):
            table = nd.table
            cols = [
                sqa.type_coerce(col, cls.sqa_type(nd.cols[col.name].dtype())).label(
                    col.name
                )
                for col in nd.table.columns
            ]
            query = Query(cols)
            sqa_col = {
                nd.cols[table_col.name]._uuid: col
                for table_col, col in zip(nd.table.columns, cols, strict=True)
            }

        if isinstance(nd, verbs.Verb):
            # decrease counters (`needed_cols` is not copied)
            for node in nd.iter_col_nodes():
                if isinstance(node, Col):
                    cnt = needed_cols.get(node._uuid)
                    if cnt == 1:
                        del needed_cols[node._uuid]
                    else:
                        needed_cols[node._uuid] = cnt - 1

        return table, query, sqa_col

    @classmethod
    def check_for_subquery(
        cls,
        nd: verbs.Verb,
        table: sqa.Table,
        sqa_col: dict[UUID, sqa.Label],
        query: Query,
        needed_cols: dict[UUID, int],
        *,
        child: verbs.Verb | None = None,
    ) -> tuple[sqa.Table, Query, dict[UUID, sqa.Label]]:
        if child is None:
            child = nd.child

        if (
            (
                isinstance(
                    nd,
                    verbs.Filter
                    | verbs.Summarize
                    | verbs.Arrange
                    | verbs.GroupBy
                    | verbs.Join,
                )
                and query.limit is not None
            )
            or (
                isinstance(nd, verbs.Mutate)
                and any(
                    any(
                        col.ftype(agg_is_window=True) in (Ftype.WINDOW, Ftype.AGGREGATE)
                        for col in fn.iter_subtree()
                        if isinstance(col, Col)
                    )
                    for fn in nd.iter_col_nodes()
                    if (
                        isinstance(fn, ColFn)
                        and fn.op.ftype in (Ftype.AGGREGATE, Ftype.WINDOW)
                    )
                )
            )
            or (
                isinstance(nd, verbs.Filter)
                and any(
                    col.ftype(agg_is_window=True) == Ftype.WINDOW
                    for col in nd.iter_col_nodes()
                    if isinstance(col, Col)
                )
            )
            or (
                isinstance(nd, verbs.Summarize)
                and (
                    (
                        bool(query.group_by)
                        and set(query.group_by) != set(query.partition_by)
                    )
                    or any(
                        (
                            node.ftype(agg_is_window=False)
                            in (Ftype.WINDOW, Ftype.AGGREGATE)
                        )
                        for node in nd.iter_col_nodes()
                        if isinstance(node, Col)
                    )
                )
            )
            or (isinstance(nd, verbs.Join) and query.group_by)
        ):
            if not isinstance(child, verbs.Alias):
                raise SubqueryError(
                    f"forbidden subquery required during compilation of `{repr(nd)}`\n"
                    "hint: If you are sure you want to do a subquery, put an "
                    "`>> alias()` before this verb. On the other hand, if you want to "
                    "write out the table of the subquery, put `>> materialize()` "
                    "before this verb."
                )

            if needed_cols.keys().isdisjoint(sqa_col.keys()):
                # We cannot select zero columns from a subquery. This happens when the
                # user only 0-ary functions after the subquery, e.g. `count`.
                needed_cols[next(iter(sqa_col.keys()))] = 1

            # We only want to select those columns that (1) the user uses in some
            # expression later or (2) are present in the final selection.

            orig_select = query.select
            query.select = []
            cnt = dict()
            name_in_subquery = dict()

            # resolve potential column name collisions in the subquery
            for uid in needed_cols.keys():
                if uid in sqa_col:
                    name = sqa_col[uid].name
                    if c := cnt.get(name):
                        name_in_subquery[uid] = f"{name}_{c}"
                        cnt[name] = c + 1
                    else:
                        name_in_subquery[uid] = name
                        cnt[name] = 1
                    query.select.append(sqa.Label(name_in_subquery[uid], sqa_col[uid]))

            table = cls.compile_query(table, query).subquery()
            sqa_col.update(
                {
                    uid: sqa.label(
                        sqa_col[uid].name, table.columns.get(name_in_subquery[uid])
                    )
                    for uid in needed_cols.keys()
                    if uid in sqa_col
                }
            )

            # rewire col refs to the subquery
            query = Query(
                select=[
                    sqa.Label(lb.name, col)
                    for lb in orig_select
                    if (col := table.columns.get(lb.name)) is not None
                ],
                partition_by=[
                    sqa.Label(lb.name, col)
                    for lb in query.partition_by
                    if (col := table.columns.get(lb.name)) is not None
                ],
            )

        return table, query, sqa_col

    # TODO: we shouldn't need these
    @classmethod
    def sqa_type(cls, pdt_type: Dtype) -> type[sqa.types.TypeEngine]:
        return pdt_type.to_sql()

    @classmethod
    def pdt_type(cls, sqa_type: sqa.types.TypeEngine) -> Dtype:
        return Dtype.from_sql(sqa_type)


@dataclasses.dataclass(slots=True)
class Query:
    select: list[sqa.Label]
    partition_by: list[sqa.Label] = dataclasses.field(default_factory=list)
    group_by: list[sqa.Label] = dataclasses.field(default_factory=list)
    where: list[sqa.ColumnElement] = dataclasses.field(default_factory=list)
    having: list[sqa.ColumnElement] = dataclasses.field(default_factory=list)
    order_by: list[sqa.UnaryExpression] = dataclasses.field(default_factory=list)
    limit: int | None = None
    offset: int | None = None


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
def create_aliases(nd: AstNode, num_occurrences: dict[str, int]) -> dict[str, int]:
    if isinstance(nd, verbs.Verb):
        num_occurrences = create_aliases(nd.child, num_occurrences)

        if isinstance(nd, verbs.Join):
            num_occurrences = create_aliases(nd.right, num_occurrences)

    elif isinstance(nd, TableImpl):
        table_name = nd.table.name
        if cnt := num_occurrences.get(table_name):
            nd.table = nd.table.alias(f"{table_name}__{cnt}")
        else:
            # always set alias to shorten queries with schemas
            nd.table = nd.table.alias(table_name)
            cnt = 0
        num_occurrences[table_name] = cnt + 1

    else:
        raise AssertionError

    return num_occurrences


def get_engine(nd: AstNode) -> sqa.Engine:
    if isinstance(nd, verbs.Verb):
        engine = get_engine(nd.child)

        if isinstance(nd, verbs.Join):
            right_engine = get_engine(nd.right)
            if engine.url != right_engine.url:
                raise NotImplementedError  # TODO: find some good error for this

    else:
        assert isinstance(nd, SqlImpl)
        engine = nd.engine

    return engine


with SqlImpl.impl_store.impl_manager as impl:
    if sqa.__version__ < "2":

        @impl(ops.floordiv)
        def _floordiv(lhs, rhs):
            return sqa.cast(lhs / rhs, sqa.Integer())

    else:

        @impl(ops.floordiv)
        def _floordiv(lhs, rhs):
            return lhs // rhs

    @impl(ops.pow)
    def _pow(lhs, rhs):
        return_type = sqa.Double()
        if isinstance(lhs.type, sqa.Numeric) and isinstance(rhs.type, sqa.Numeric):
            return_type = sqa.Numeric()
        return sqa.func.POW(lhs, rhs, type_=return_type)

    @impl(ops.bool_xor)
    def _xor(lhs, rhs):
        return lhs != rhs

    @impl(ops.pos)
    def _pos(x):
        return x

    @impl(ops.abs)
    def _abs(x):
        return sqa.func.ABS(x, type_=x.type)

    @impl(ops.round)
    def _round(x, decimals=0):
        return sqa.func.ROUND(x, decimals, type_=x.type)

    @impl(ops.is_in)
    def _is_in(x, *values):
        return x.in_(v for v in values)

    @impl(ops.is_null)
    def _is_null(x):
        return x.is_(sqa.null())

    @impl(ops.is_not_null)
    def _is_not_null(x):
        return x.is_not(sqa.null())

    @impl(ops.str_strip)
    def _str_strip(x):
        return sqa.func.TRIM(x, type_=x.type)

    @impl(ops.str_len)
    def _str_length(x):
        return sqa.func.LENGTH(x, type_=sqa.Integer())

    @impl(ops.str_upper)
    def _upper(x):
        return sqa.func.UPPER(x, type_=x.type)

    @impl(ops.str_lower)
    def _lower(x):
        return sqa.func.LOWER(x, type_=x.type)

    @impl(ops.str_replace_all)
    def _str_replace_all(x, y, z):
        return sqa.func.REPLACE(x, y, z, type_=x.type)

    @impl(ops.str_starts_with)
    def _str_starts_with(x, y):
        return x.startswith(y, autoescape=True)

    @impl(ops.str_ends_with)
    def _str_ends_with(x, y):
        return x.endswith(y, autoescape=True)

    @impl(ops.str_contains)
    def _str_contains(x, y):
        return x.contains(y, autoescape=True)

    @impl(ops.str_slice)
    def _str_slice(x, offset, length):
        # SQL has 1-indexed strings but we do it 0-indexed
        return sqa.func.SUBSTR(x, offset + 1, length)

    @impl(ops.dt_year)
    def _dt_year(x):
        return sqa.extract("year", x)

    @impl(ops.dt_month)
    def _dt_month(x):
        return sqa.extract("month", x)

    @impl(ops.dt_day)
    def _dt_day(x):
        return sqa.extract("day", x)

    @impl(ops.dt_hour)
    def _dt_hour(x):
        return sqa.extract("hour", x)

    @impl(ops.dt_minute)
    def _dt_minute(x):
        return sqa.extract("minute", x)

    @impl(ops.dt_second)
    def _dt_second(x):
        return sqa.extract("second", x)

    @impl(ops.dt_millisecond)
    def _dt_millisecond(x):
        return sqa.extract("milliseconds", x) % 1000

    @impl(ops.dt_day_of_week)
    def _day_of_week(x):
        return sqa.extract("dow", x)

    @impl(ops.dt_day_of_year)
    def _day_of_year(x):
        return sqa.extract("doy", x)

    @impl(ops.horizontal_max)
    def _horizontal_max(*x):
        return sqa.func.GREATEST(*x)

    @impl(ops.horizontal_min)
    def _horizontal_min(*x):
        return sqa.func.LEAST(*x)

    @impl(ops.mean)
    def _mean(x):
        type_ = sqa.Numeric()
        if isinstance(x.type, sqa.Float):
            type_ = sqa.Double()
        return sqa.func.AVG(x, type_=type_)

    @impl(ops.min)
    def _min(x):
        return sqa.func.min(x)

    @impl(ops.max)
    def _max(x):
        return sqa.func.max(x)

    @impl(ops.sum)
    def _sum(x):
        return sqa.func.sum(x)

    @impl(ops.any)
    def _any(x):
        return sqa.func.max(x)

    @impl(ops.all)
    def _all(x):
        return sqa.func.min(x)

    @impl(ops.count)
    def _count(x=None):
        return sqa.func.count(x)

    @impl(ops.count_star)
    def _len():
        return sqa.func.count()

    @impl(ops.shift)
    def _shift(x, by, empty_value=None):
        if by >= 0:
            return sqa.func.LAG(x, by, empty_value, type_=x.type)
        if by < 0:
            return sqa.func.LEAD(x, -by, empty_value, type_=x.type)

    @impl(ops.row_number)
    def _row_number():
        return sqa.func.ROW_NUMBER(type_=sqa.Integer())

    @impl(ops.rank)
    def _rank():
        return sqa.func.rank()

    @impl(ops.dense_rank)
    def _dense_rank():
        return sqa.func.dense_rank()

    @impl(ops.exp)
    def _exp(x):
        return sqa.func.exp(x)

    @impl(ops.log)
    def _log(x):
        return sqa.func.ln(x)

    @impl(ops.floor)
    def _floor(x):
        return sqa.func.floor(x)

    @impl(ops.ceil)
    def _ceil(x):
        return sqa.func.ceil(x)

    @impl(ops.str_to_datetime)
    def _str_to_datetime(x):
        return sqa.cast(x, sqa.DateTime)

    @impl(ops.str_to_date)
    def _str_to_date(x):
        return sqa.cast(x, sqa.Date)

    @impl(ops.is_inf)
    def _is_inf(x, *, _Impl):
        return x == _Impl.inf()

    @impl(ops.is_not_inf)
    def _is_not_inf(x, *, _Impl):
        return x != _Impl.inf()

    @impl(ops.coalesce)
    def _coalesce(*x):
        return sqa.func.coalesce(*x)

    @impl(ops.str_join)
    def _str_join(x, delim):
        return sqa.func.string_agg(x, delim)

    @impl(ops.fill_null)
    def _fill_null(x, y):
        return sqa.func.coalesce(x, y)
