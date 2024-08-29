from __future__ import annotations

import functools
import inspect
import itertools
import operator as py_operator
import uuid
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal

import polars as pl
import sqlalchemy as sa
from sqlalchemy import sql

from pydiverse.transform import ops
from pydiverse.transform._typing import ImplT
from pydiverse.transform.core import dtypes
from pydiverse.transform.core.expressions import (
    Column,
    LiteralColumn,
    SymbolicExpression,
    iterate_over_expr,
)
from pydiverse.transform.core.expressions.translator import TypedValue
from pydiverse.transform.core.table_impl import AbstractTableImpl, ColumnMetaData
from pydiverse.transform.core.util import OrderingDescriptor, translate_ordering
from pydiverse.transform.errors import AlignmentError, FunctionTypeError
from pydiverse.transform.ops import OPType

if TYPE_CHECKING:
    from pydiverse.transform.core.registry import TypedOperatorImpl


class SQLTableImpl(AbstractTableImpl):
    """SQL backend

    Attributes:
        tbl: The underlying SQLAlchemy table object.
        engine: The SQLAlchemy engine.
        sql_columns: A dict mapping from uuids to SQLAlchemy column objects
            (only those contained in `tbl`).

        alignment_hash: A hash value that allows checking if two tables are
            'aligned'. In the case of SQL this means that two tables NUST NOT
            share the same alignment hash unless they were derived from the
            same table(s) and are guaranteed to have the same number of columns
            in the same order. In other words: Two tables MUST only have the
            same alignment hash if a literal column derived from one of them
            can be used by the other table and produces the same result.
    """

    __registered_dialects: dict[str, type[SQLTableImpl]] = {}
    _dialect_name: str

    def __new__(cls, *args, **kwargs):
        if cls != SQLTableImpl or (not args and not kwargs):
            return super().__new__(cls)

        signature = inspect.signature(cls.__init__)
        engine = signature.bind(None, *args, **kwargs).arguments["engine"]

        # If calling SQLTableImpl(engine), then we want to dynamically instantiate
        # the correct dialect specific subclass based on the engine dialect.
        if isinstance(engine, str):
            dialect = sa.engine.make_url(engine).get_dialect().name
        else:
            dialect = engine.dialect.name

        dialect_specific_cls = SQLTableImpl.__registered_dialects.get(dialect, cls)
        return super(SQLTableImpl, dialect_specific_cls).__new__(dialect_specific_cls)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Whenever a new subclass if SQLTableImpl is defined, it must contain the
        # `_dialect_name` attribute. This allows us to dynamically instantiate it
        # when calling SQLTableImpl(engine) based on the dialect name found
        # in the engine url (see __new__).
        dialect_name = getattr(cls, "_dialect_name", None)
        if dialect_name is None:
            raise ValueError(
                "All subclasses of SQLTableImpl must have a `_dialect_name` attribute."
                f" But {cls.__name__}._dialect_name is None."
            )

        if dialect_name in SQLTableImpl.__registered_dialects:
            warnings.warn(
                f"Already registered a SQLTableImpl for dialect {dialect_name}"
            )
        SQLTableImpl.__registered_dialects[dialect_name] = cls

    def __init__(
        self,
        engine: sa.Engine | str,
        table,
        _dtype_hints: dict[str, dtypes.DType] = None,
    ):
        self.engine = sa.create_engine(engine) if isinstance(engine, str) else engine
        tbl = self._create_table(table, self.engine)

        columns = {
            col.name: Column(
                name=col.name,
                table=self,
                dtype=self._get_dtype(col, hints=_dtype_hints),
            )
            for col in tbl.columns
        }

        self.replace_tbl(tbl, columns)
        super().__init__(name=self.tbl.name, columns=columns)

    def is_aligned_with(self, col: Column | LiteralColumn) -> bool:
        if isinstance(col, Column):
            if not isinstance(col.table, type(self)):
                return False
            return col.table.alignment_hash == self.alignment_hash

        if isinstance(col, LiteralColumn):
            return all(
                self.is_aligned_with(atom)
                for atom in iterate_over_expr(col.expr, expand_literal_col=True)
                if isinstance(atom, Column)
            )

        raise ValueError

    @classmethod
    def _html_repr_expr(cls, expr):
        if isinstance(expr, sa.sql.ColumnElement):
            return str(expr.compile(compile_kwargs={"literal_binds": True}))
        return super()._html_repr_expr(expr)

    @staticmethod
    def _create_table(tbl, engine=None):
        """Return a sa.Table

        :param tbl: a sa.Table or string of form 'table_name'
            or 'schema_name.table_name'.
        """
        if isinstance(tbl, sa.sql.FromClause):
            return tbl

        if not isinstance(tbl, str):
            raise ValueError(f"tbl must be a sqlalchemy Table or string, but was {tbl}")

        schema, table_name = tbl.split(".") if "." in tbl else [None, tbl]
        return sa.Table(
            table_name,
            sa.MetaData(),
            schema=schema,
            autoload_with=engine,
        )

    @staticmethod
    def _get_dtype(
        col: sa.Column, hints: dict[str, dtypes.DType] = None
    ) -> dtypes.DType:
        """Determine the dtype of a column.

        :param col: The sqlalchemy column object.
        :param hints: In some situations sqlalchemy can't determine the dtype of
            a column. Instead of throwing an exception we can use these type
            hints as a fallback.
        :return: Appropriate dtype string.
        """

        type_ = col.type
        if isinstance(type_, sa.Integer):
            return dtypes.Int()
        if isinstance(type_, sa.Numeric):
            return dtypes.Float()
        if isinstance(type_, sa.String):
            return dtypes.String()
        if isinstance(type_, sa.Boolean):
            return dtypes.Bool()
        if isinstance(type_, sa.DateTime):
            return dtypes.DateTime()
        if isinstance(type_, sa.Date):
            return dtypes.Date()
        if isinstance(type_, sa.Interval):
            return dtypes.Duration()
        if isinstance(type_, sa.Time):
            raise NotImplementedError("Unsupported type: Time")

        if hints is not None:
            if dtype := hints.get(col.name):
                return dtype

        raise NotImplementedError(f"Unsupported type: {type_}")

    def replace_tbl(self, new_tbl, columns: dict[str:Column]):
        if isinstance(new_tbl, sql.Select):
            # noinspection PyNoneFunctionAssignment
            new_tbl = new_tbl.subquery()

        self.tbl = new_tbl
        self.alignment_hash = generate_alignment_hash()

        self.sql_columns = {
            col.uuid: self.tbl.columns[col.name] for col in columns.values()
        }  # from uuid to sqlalchemy column

        if hasattr(self, "cols"):
            # TODO: Clean up... This feels a bit hacky
            for col in columns.values():
                self.cols[col.uuid] = ColumnMetaData.from_expr(col.uuid, col, self)
        if hasattr(self, "intrinsic_grouped_by"):
            self.intrinsic_grouped_by.clear()

        self.joins: list[JoinDescriptor] = []
        self.wheres: list[SymbolicExpression] = []
        self.having: list[SymbolicExpression] = []
        self.order_bys: list[OrderingDescriptor] = []
        self.limit_offset: tuple[int, int] | None = None

    def build_select(self) -> sql.Select:
        # Validate current state
        if len(self.selects) == 0:
            raise ValueError("Can't execute a SQL query without any SELECT statements.")

        # Start building query
        select = self.tbl.select()

        # `select_from` is required if no table is explicitly referenced
        # inside the SELECT. e.g. `SELECT COUNT(*) AS count`
        select = select.select_from(self.tbl)

        # FROM
        select = self._build_select_from(select)

        # WHERE
        select = self._build_select_where(select)

        # GROUP BY
        select = self._build_select_group_by(select)

        # HAVING
        select = self._build_select_having(select)

        # LIMIT / OFFSET
        select = self._build_select_limit_offset(select)

        # SELECT
        select = self._build_select_select(select)

        # ORDER BY
        select = self._build_select_order_by(select)

        return select

    def _build_select_from(self, select):
        for join in self.joins:
            compiled, _ = self.compiler.translate(join.on, verb="join")
            on = compiled(self.sql_columns)

            select = select.join(
                join.right.tbl,
                onclause=on,
                isouter=join.how != "inner",
                full=join.how == "outer",
            )

        return select

    def _build_select_where(self, select):
        if not self.wheres:
            return select

        # Combine wheres using ands
        combined_where = functools.reduce(
            py_operator.and_, map(SymbolicExpression, self.wheres)
        )._
        compiled, where_dtype = self.compiler.translate(combined_where, verb="filter")
        assert isinstance(where_dtype, dtypes.Bool)
        where = compiled(self.sql_columns)
        return select.where(where)

    def _build_select_group_by(self, select):
        if not self.intrinsic_grouped_by:
            return select

        compiled_gb, group_by_dtypes = zip(
            *(
                self.compiler.translate(group_by, verb="group_by")
                for group_by in self.intrinsic_grouped_by
            )
        )
        group_bys = (compiled(self.sql_columns) for compiled in compiled_gb)
        return select.group_by(*group_bys)

    def _build_select_having(self, select):
        if not self.having:
            return select

        # Combine havings using ands
        combined_having = functools.reduce(
            py_operator.and_, map(SymbolicExpression, self.having)
        )._
        compiled, having_dtype = self.compiler.translate(combined_having, verb="filter")
        assert isinstance(having_dtype, dtypes.Bool)
        having = compiled(self.sql_columns)
        return select.having(having)

    def _build_select_limit_offset(self, select):
        if self.limit_offset is None:
            return select

        limit, offset = self.limit_offset
        return select.limit(limit).offset(offset)

    def _build_select_select(self, select):
        # Convert self.selects to SQLAlchemy Expressions
        s = []
        for name, uuid_ in self.selected_cols():
            sql_col = self.cols[uuid_].compiled(self.sql_columns)
            if not isinstance(sql_col, sa.sql.ColumnElement):
                sql_col = sa.literal(sql_col)
            s.append(sql_col.label(name))
        return select.with_only_columns(*s)

    def _build_select_order_by(self, select):
        if not self.order_bys:
            return select

        o = []
        for o_by in self.order_bys:
            compiled, _ = self.compiler.translate(o_by.order, verb="arrange")
            col = compiled(self.sql_columns)
            o.extend(self._order_col(col, o_by))

        return select.order_by(*o)

    #### Verb Operations ####

    def preverb_hook(self, verb: str, *args, **kwargs) -> None:
        def has_any_ftype_cols(ftypes: OPType | tuple[OPType, ...], cols: Iterable):
            if isinstance(ftypes, OPType):
                ftypes = (ftypes,)
            return any(
                self.cols[c.uuid].ftype in ftypes
                for v in cols
                for c in iterate_over_expr(self.resolve_lambda_cols(v))
                if isinstance(c, Column)
            )

        requires_subquery = False
        clear_order = False

        if self.limit_offset is not None:
            # The LIMIT / TOP clause is executed at the very end of the query.
            # This means we must create a subquery for any verb that modifies
            # the rows.
            if verb in (
                "join",
                "filter",
                "arrange",
                "group_by",
                "summarise",
            ):
                requires_subquery = True

        if verb == "mutate":
            # Window functions can't be nested, thus a subquery is required
            requires_subquery |= has_any_ftype_cols(OPType.WINDOW, kwargs.values())
        elif verb == "filter":
            # Window functions aren't allowed in where clause
            requires_subquery |= has_any_ftype_cols(OPType.WINDOW, args)
        elif verb == "summarise":
            # The result of the aggregate is always ordered according to the
            # grouping columns. We must clear the order_bys so that the order
            # is consistent with eager execution. We can do this because aggregate
            # functions are independent of the order.
            clear_order = True

            # If the grouping level is different from the grouping level of the
            # tbl object, or if on of the input columns is a window or aggregate
            # function, we must make a subquery.
            requires_subquery |= (
                bool(self.intrinsic_grouped_by)
                and self.grouped_by != self.intrinsic_grouped_by
            )
            requires_subquery |= has_any_ftype_cols(
                (OPType.AGGREGATE, OPType.WINDOW), kwargs.values()
            )

        # TODO: It would be nice if this could be done without having to select all
        #       columns. As a potential challenge for the hackathon I propose a mean
        #       of even creating the subqueries lazyly. This means that we could
        #       perform some kind of query optimization before submitting the actual
        #       query. Eg: Instead of selecting all possible columns, only select
        #       those that actually get used.
        if requires_subquery:
            columns = {
                name: self.cols[uuid].as_column(name, self)
                for name, uuid in self.named_cols.fwd.items()
            }

            original_selects = self.selects.copy()
            self.selects |= columns.keys()
            subquery = self.build_select()

            self.replace_tbl(subquery, columns)
            self.selects = original_selects

        if clear_order:
            self.order_bys.clear()

    def alias(self, name=None):
        if name is None:
            suffix = format(uuid.uuid1().int % 0x7FFFFFFF, "X")
            name = f"{self.name}_{suffix}"

        # TODO: If the table has not been modified, a simple `.alias()`
        #       would produce nicer queries.
        subquery = self.build_select().subquery(name=name)
        # In some situations sqlalchemy fails to determine the datatype of a column.
        # To circumvent this, we can pass on the information we know.
        dtype_hints = {
            name: self.cols[self.named_cols.fwd[name]].dtype for name in self.selects
        }

        return self.__class__(self.engine, subquery, _dtype_hints=dtype_hints)

    def collect(self):
        select = self.build_select()
        with self.engine.connect() as conn:
            try:
                # TODO: check for which pandas versions this is needed:
                # Temporary fix for pandas bug (https://github.com/pandas-dev/pandas/issues/35484)
                # Taken from siuba
                from pandas.io import sql as _pd_sql

                class _FixedSqlDatabase(_pd_sql.SQLDatabase):
                    def execute(self, *args, **kwargs):
                        return self.connectable.execute(*args, **kwargs)

                sql_db = _FixedSqlDatabase(conn)
                result = sql_db.read_sql(select).convert_dtypes()
            except AttributeError:
                import pandas as pd

                result = pd.read_sql_query(select, con=conn)

        # Add metadata
        result.attrs["name"] = self.name
        return result

    def export(self):
        with self.engine.connect() as conn:
            if isinstance(self, DuckDBTableImpl):
                result = pl.read_database(self.build_query(), connection=conn)
            else:
                result = pl.read_database(self.build_select(), connection=conn)
        return result

    def build_query(self) -> str:
        query = self.build_select()
        return str(
            query.compile(
                dialect=self.engine.dialect, compile_kwargs={"literal_binds": True}
            )
        )

    def join(
        self,
        right: SQLTableImpl,
        on: SymbolicExpression,
        how: Literal["inner", "left", "outer"],
        *,
        validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    ):
        self.alignment_hash = generate_alignment_hash()

        # If right has joins already, merging them becomes extremely difficult
        # This is because the ON clauses could contain NULL checks in which case
        # the joins aren't associative anymore.
        if right.joins:
            raise ValueError(
                "Can't automatically combine joins if the right side already contains a"
                " JOIN clause."
            )

        if right.limit_offset is not None:
            raise ValueError(
                "The right table can't be sliced when performing a join."
                " Wrap the right side in a subquery to fix this."
            )

        # TODO: Handle GROUP BY and SELECTS on left / right side

        # Combine the WHERE clauses
        if how == "inner":
            # Inner Join: The WHERES can be combined
            self.wheres.extend(right.wheres)
        elif how == "left":
            # WHERES from right must go into the ON clause
            on = reduce(py_operator.and_, (on, *right.wheres))
        elif how == "outer":
            # For outer joins, the WHERE clause can't easily be merged.
            # The best solution for now is to move them into a subquery.
            if self.wheres:
                raise ValueError(
                    "Filters can't precede outer joins. Wrap the left side in a"
                    " subquery to fix this."
                )
            if right.wheres:
                raise ValueError(
                    "Filters can't precede outer joins. Wrap the right side in a"
                    " subquery to fix this."
                )

        if validate != "m:m":
            warnings.warn("SQL table backend ignores join validation argument.")

        descriptor = JoinDescriptor(right, on, how)
        self.joins.append(descriptor)

        self.sql_columns.update(right.sql_columns)

    def filter(self, *args):
        self.alignment_hash = generate_alignment_hash()

        if self.intrinsic_grouped_by:
            for arg in args:
                # If a condition involves only grouping columns, it can be
                # moved into the wheres instead of the havings.
                only_grouping_cols = all(
                    col in self.intrinsic_grouped_by
                    for col in iterate_over_expr(arg, expand_literal_col=True)
                    if isinstance(col, Column)
                )

                if only_grouping_cols:
                    self.wheres.append(arg)
                else:
                    self.having.append(arg)
        else:
            self.wheres.extend(args)

    def arrange(self, ordering):
        self.alignment_hash = generate_alignment_hash()

        # Merge order bys and remove duplicate columns
        order_bys = []
        order_by_columns = set()
        for o_by in ordering + self.order_bys:
            if o_by.order in order_by_columns:
                continue
            order_bys.append(o_by)
            order_by_columns.add(o_by.order)

        self.order_bys = order_bys

    def summarise(self, **kwargs):
        self.alignment_hash = generate_alignment_hash()

    def slice_head(self, n: int, offset: int):
        if self.limit_offset is None:
            self.limit_offset = (n, offset)
        else:
            old_n, old_o = self.limit_offset
            self.limit_offset = (min(abs(old_n - offset), n), old_o + offset)

    #### EXPRESSIONS ####

    def _order_col(
        self, col: sa.SQLColumnExpression, ordering: OrderingDescriptor
    ) -> list[sa.SQLColumnExpression]:
        col = col.asc() if ordering.asc else col.desc()
        col = col.nullsfirst() if ordering.nulls_first else col.nullslast()
        return [col]

    class ExpressionCompiler(
        AbstractTableImpl.ExpressionCompiler[
            "SQLTableImpl",
            TypedValue[Callable[[dict[uuid.UUID, sa.Column]], sql.ColumnElement]],
        ]
    ):
        def _translate_col(self, col, **kwargs):
            # Can either be a base SQL column, or a reference to an expression
            if col.uuid in self.backend.sql_columns:

                def sql_col(cols, **kw):
                    return cols[col.uuid]

                return TypedValue(sql_col, col.dtype, OPType.EWISE)

            meta_data = self.backend.cols[col.uuid]
            return TypedValue(meta_data.compiled, meta_data.dtype, meta_data.ftype)

        def _translate_literal_col(self, expr, **kwargs):
            if not self.backend.is_aligned_with(expr):
                raise AlignmentError(
                    "Literal column isn't aligned with this table. "
                    f"Literal Column: {expr}"
                )

            def sql_col(cols, **kw):
                return expr.typed_value.value

            return TypedValue(sql_col, expr.typed_value.dtype, expr.typed_value.ftype)

        def _translate_function(
            self, implementation, op_args, context_kwargs, *, verb=None, **kwargs
        ):
            value = self._translate_function_value(
                implementation,
                op_args,
                context_kwargs,
                verb=verb,
                **kwargs,
            )
            operator = implementation.operator

            if operator.ftype == OPType.AGGREGATE and verb == "mutate":
                # Aggregate function in mutate verb -> window function
                over_value = self.over_clause(value, implementation, context_kwargs)
                ftype = self.backend._get_op_ftype(
                    op_args, operator, OPType.WINDOW, strict=True
                )
                return TypedValue(over_value, implementation.rtype, ftype)

            elif operator.ftype == OPType.WINDOW:
                if verb != "mutate":
                    raise FunctionTypeError(
                        "Window function are only allowed inside a mutate."
                    )

                over_value = self.over_clause(value, implementation, context_kwargs)
                ftype = self.backend._get_op_ftype(op_args, operator, strict=True)
                return TypedValue(over_value, implementation.rtype, ftype)

            else:
                ftype = self.backend._get_op_ftype(op_args, operator, strict=True)
                return TypedValue(value, implementation.rtype, ftype)

        def _translate_function_value(
            self,
            implementation: TypedOperatorImpl,
            op_args: list,
            context_kwargs: dict,
            *,
            verb=None,
            **kwargs,
        ):
            impl_dtypes = implementation.impl.signature.args
            if implementation.impl.signature.is_vararg:
                impl_dtypes = itertools.chain(
                    impl_dtypes[:-1],
                    itertools.repeat(impl_dtypes[-1]),
                )

            def value(cols, *, variant=None, internal_kwargs=None, **kw):
                args = []
                for arg, dtype in zip(op_args, impl_dtypes):
                    if dtype.const:
                        args.append(arg.value(cols, as_sql_literal=False))
                    else:
                        args.append(arg.value(cols))

                kwargs = {
                    "_tbl": self.backend,
                    "_verb": verb,
                    **(internal_kwargs or {}),
                }

                if variant is not None:
                    if variant_impl := implementation.get_variant(variant):
                        return variant_impl(*args, **kwargs)

                return implementation(*args, **kwargs)

            return value

        def _translate_case(self, expr, switching_on, cases, default, **kwargs):
            def value(*args, **kw):
                default_ = default.value(*args, **kwargs)

                if switching_on is not None:
                    switching_on_ = switching_on.value(*args, **kwargs)
                    return sa.case(
                        {
                            cond.value(*args, **kw): val.value(*args, **kw)
                            for cond, val in cases
                        },
                        value=switching_on_,
                        else_=default_,
                    )

                return sa.case(
                    *(
                        (cond.value(*args, **kw), val.value(*args, **kw))
                        for cond, val in cases
                    ),
                    else_=default_,
                )

            result_dtype, result_ftype = self._translate_case_common(
                expr, switching_on, cases, default, **kwargs
            )
            return TypedValue(value, result_dtype, result_ftype)

        def _translate_literal_value(self, expr):
            def literal_func(*args, as_sql_literal=True, **kwargs):
                if as_sql_literal:
                    return sa.literal(expr)
                return expr

            return literal_func

        def over_clause(
            self,
            value: Callable,
            implementation: TypedOperatorImpl,
            context_kwargs: dict,
        ):
            operator = implementation.operator
            if operator.ftype not in (OPType.AGGREGATE, OPType.WINDOW):
                raise FunctionTypeError

            wants_order_by = operator.ftype == OPType.WINDOW

            # PARTITION BY
            grouping = context_kwargs.get("partition_by")
            if grouping is not None:
                grouping = [self.backend.resolve_lambda_cols(col) for col in grouping]
            else:
                grouping = self.backend.grouped_by

            compiled_pb = tuple(self.translate(col).value for col in grouping)

            # ORDER BY
            def order_by_clause_generator(ordering: OrderingDescriptor):
                compiled, _ = self.translate(ordering.order)

                def clause(*args, **kwargs):
                    col = compiled(*args, **kwargs)
                    return self.backend._order_col(col, ordering)

                return clause

            if wants_order_by:
                arrange = context_kwargs.get("arrange")
                if not arrange:
                    raise TypeError("Missing 'arrange' argument.")

                ordering = translate_ordering(self.backend, arrange)
                compiled_ob = [order_by_clause_generator(o_by) for o_by in ordering]

            # New value callable
            def over_value(*args, **kwargs):
                pb = sql.expression.ClauseList(
                    *(compiled(*args, **kwargs) for compiled in compiled_pb)
                )
                ob = (
                    sql.expression.ClauseList(
                        *(
                            clause
                            for compiled in compiled_ob
                            for clause in compiled(*args, **kwargs)
                        )
                    )
                    if wants_order_by
                    else None
                )

                # Some operators need to further modify the OVER expression
                # To do this, we allow registering a variant called "window"
                if implementation.has_variant("window"):
                    return value(
                        *args,
                        variant="window",
                        internal_kwargs={
                            "_window_partition_by": pb,
                            "_window_order_by": ob,
                        },
                        **kwargs,
                    )

                # If now window variant has been defined, just apply generic OVER clause
                return value(*args, **kwargs).over(
                    partition_by=pb,
                    order_by=ob,
                )

            return over_value

    class AlignedExpressionEvaluator(
        AbstractTableImpl.AlignedExpressionEvaluator[TypedValue[sql.ColumnElement]]
    ):
        def translate(self, expr, check_alignment=True, **kwargs):
            if check_alignment:
                alignment_hashes = {
                    col.table.alignment_hash
                    for col in iterate_over_expr(expr, expand_literal_col=True)
                    if isinstance(col, Column)
                }
                if len(alignment_hashes) >= 2:
                    raise AlignmentError(
                        "Expression contains columns from different tables that aren't"
                        " aligned."
                    )

            return super().translate(expr, check_alignment=check_alignment, **kwargs)

        def _translate_col(self, col, **kwargs):
            backend = col.table
            if col.uuid in backend.sql_columns:
                sql_col = backend.sql_columns[col.uuid]
                return TypedValue(sql_col, col.dtype)

            meta_data = backend.cols[col.uuid]
            return TypedValue(
                meta_data.compiled(backend.sql_columns),
                meta_data.dtype,
                meta_data.ftype,
            )

        def _translate_literal_col(self, expr, **kwargs):
            assert issubclass(expr.backend, SQLTableImpl)
            return expr.typed_value

        def _translate_function(
            self, implementation, op_args, context_kwargs, **kwargs
        ):
            # Aggregate function -> window function
            value = implementation(*(arg.value for arg in op_args))
            operator = implementation.operator
            override_ftype = (
                OPType.WINDOW if operator.ftype == OPType.AGGREGATE else None
            )
            ftype = SQLTableImpl._get_op_ftype(
                op_args, operator, override_ftype, strict=True
            )

            if operator.ftype == OPType.AGGREGATE:
                value = value.over()
            if operator.ftype == OPType.WINDOW:
                raise NotImplementedError("How to handle window functions?")

            return TypedValue(value, implementation.rtype, ftype)


@dataclass
class JoinDescriptor(Generic[ImplT]):
    __slots__ = ("right", "on", "how")

    right: ImplT
    on: Any
    how: str


def generate_alignment_hash():
    # It should be possible to have an alternative hash value that
    # is a bit more lenient -> If the same set of operations get applied
    # to a table in two different orders that produce the same table
    # object, their hash could also be equal.
    return uuid.uuid1()


#### BACKEND SPECIFIC OPERATORS ################################################


with SQLTableImpl.op(ops.FloorDiv(), check_super=False) as op:
    if sa.__version__ < "2":

        @op.auto
        def _floordiv(lhs, rhs):
            return sa.cast(lhs / rhs, sa.Integer())

    else:

        @op.auto
        def _floordiv(lhs, rhs):
            return lhs // rhs


with SQLTableImpl.op(ops.RFloorDiv(), check_super=False) as op:

    @op.auto
    def _rfloordiv(rhs, lhs):
        return _floordiv(lhs, rhs)


with SQLTableImpl.op(ops.Pow()) as op:

    @op.auto
    def _pow(lhs, rhs):
        if isinstance(lhs.type, sa.Float) or isinstance(rhs.type, sa.Float):
            type_ = sa.Double()
        elif isinstance(lhs.type, sa.Numeric) or isinstance(rhs, sa.Numeric):
            type_ = sa.Numeric()
        else:
            type_ = sa.Double()

        return sa.func.POW(lhs, rhs, type_=type_)


with SQLTableImpl.op(ops.RPow()) as op:

    @op.auto
    def _rpow(rhs, lhs):
        return _pow(lhs, rhs)


with SQLTableImpl.op(ops.Xor()) as op:

    @op.auto
    def _xor(lhs, rhs):
        return lhs != rhs


with SQLTableImpl.op(ops.RXor()) as op:

    @op.auto
    def _rxor(rhs, lhs):
        return lhs != rhs


with SQLTableImpl.op(ops.Pos()) as op:

    @op.auto
    def _pos(x):
        return x


with SQLTableImpl.op(ops.Abs()) as op:

    @op.auto
    def _abs(x):
        return sa.func.ABS(x, type_=x.type)


with SQLTableImpl.op(ops.Round()) as op:

    @op.auto
    def _round(x, decimals=0):
        return sa.func.ROUND(x, decimals, type_=x.type)


with SQLTableImpl.op(ops.IsIn()) as op:

    @op.auto
    def _isin(x, *values, _verb=None):
        if _verb == "filter":
            # In WHERE and HAVING clause, we can use the IN operator
            return x.in_(values)
        # In SELECT we must replace it with the corresponding boolean expression
        return reduce(py_operator.or_, map(lambda v: x == v, values))


with SQLTableImpl.op(ops.IsNull()) as op:

    @op.auto
    def _is_null(x):
        return x.is_(sa.null())


with SQLTableImpl.op(ops.IsNotNull()) as op:

    @op.auto
    def _is_not_null(x):
        return x.is_not(sa.null())


#### String Functions ####


with SQLTableImpl.op(ops.StrStrip()) as op:

    @op.auto
    def _str_strip(x):
        return sa.func.TRIM(x, type_=x.type)


with SQLTableImpl.op(ops.StrLen()) as op:

    @op.auto
    def _str_length(x):
        return sa.func.LENGTH(x, type_=sa.Integer())


with SQLTableImpl.op(ops.StrToUpper()) as op:

    @op.auto
    def _upper(x):
        return sa.func.UPPER(x, type_=x.type)


with SQLTableImpl.op(ops.StrToLower()) as op:

    @op.auto
    def _upper(x):
        return sa.func.LOWER(x, type_=x.type)


with SQLTableImpl.op(ops.StrReplaceAll()) as op:

    @op.auto
    def _replace(x, y, z):
        return sa.func.REPLACE(x, y, z, type_=x.type)


with SQLTableImpl.op(ops.StrStartsWith()) as op:

    @op.auto
    def _startswith(x, y):
        return x.startswith(y, autoescape=True)


with SQLTableImpl.op(ops.StrEndsWith()) as op:

    @op.auto
    def _endswith(x, y):
        return x.endswith(y, autoescape=True)


with SQLTableImpl.op(ops.StrContains()) as op:

    @op.auto
    def _contains(x, y):
        return x.contains(y, autoescape=True)


with SQLTableImpl.op(ops.StrSlice()) as op:

    @op.auto
    def _str_slice(x, offset, length):
        # SQL has 1-indexed strings but we do it 0-indexed
        return sa.func.SUBSTR(x, offset + 1, length)


#### Datetime Functions ####


with SQLTableImpl.op(ops.DtYear()) as op:

    @op.auto
    def _year(x):
        return sa.extract("year", x)


with SQLTableImpl.op(ops.DtMonth()) as op:

    @op.auto
    def _month(x):
        return sa.extract("month", x)


with SQLTableImpl.op(ops.DtDay()) as op:

    @op.auto
    def _day(x):
        return sa.extract("day", x)


with SQLTableImpl.op(ops.DtHour()) as op:

    @op.auto
    def _hour(x):
        return sa.extract("hour", x)


with SQLTableImpl.op(ops.DtMinute()) as op:

    @op.auto
    def _minute(x):
        return sa.extract("minute", x)


with SQLTableImpl.op(ops.DtSecond()) as op:

    @op.auto
    def _second(x):
        return sa.extract("second", x)


with SQLTableImpl.op(ops.DtMillisecond()) as op:

    @op.auto
    def _millisecond(x):
        return sa.extract("milliseconds", x) % 1000


with SQLTableImpl.op(ops.DtDayOfWeek()) as op:

    @op.auto
    def _day_of_week(x):
        return sa.extract("dow", x)


with SQLTableImpl.op(ops.DtDayOfYear()) as op:

    @op.auto
    def _day_of_year(x):
        return sa.extract("doy", x)


#### Generic Functions ####


with SQLTableImpl.op(ops.Greatest()) as op:

    @op.auto
    def _greatest(*x):
        # TODO: Determine return type
        return sa.func.GREATEST(*x)


with SQLTableImpl.op(ops.Least()) as op:

    @op.auto
    def _least(*x):
        # TODO: Determine return type
        return sa.func.LEAST(*x)


#### Summarising Functions ####


with SQLTableImpl.op(ops.Mean()) as op:

    @op.auto
    def _mean(x):
        type_ = sa.Numeric()
        if isinstance(x.type, sa.Float):
            type_ = sa.Double()

        return sa.func.AVG(x, type_=type_)


with SQLTableImpl.op(ops.Min()) as op:

    @op.auto
    def _min(x):
        return sa.func.min(x)


with SQLTableImpl.op(ops.Max()) as op:

    @op.auto
    def _max(x):
        return sa.func.max(x)


with SQLTableImpl.op(ops.Sum()) as op:

    @op.auto
    def _sum(x):
        return sa.func.sum(x)


with SQLTableImpl.op(ops.Any()) as op:

    @op.auto
    def _any(x, *, _window_partition_by=None, _window_order_by=None):
        return sa.func.coalesce(sa.func.max(x), sa.false())

    @op.auto(variant="window")
    def _any(x, *, _window_partition_by=None, _window_order_by=None):
        return sa.func.coalesce(
            sa.func.max(x).over(
                partition_by=_window_partition_by,
                order_by=_window_order_by,
            ),
            sa.false(),
        )


with SQLTableImpl.op(ops.All()) as op:

    @op.auto
    def _all(x):
        return sa.func.coalesce(sa.func.min(x), sa.false())

    @op.auto(variant="window")
    def _all(x, *, _window_partition_by=None, _window_order_by=None):
        return sa.func.coalesce(
            sa.func.min(x).over(
                partition_by=_window_partition_by,
                order_by=_window_order_by,
            ),
            sa.false(),
        )


with SQLTableImpl.op(ops.Count()) as op:

    @op.auto
    def _count(x=None):
        if x is None:
            # Get the number of rows
            return sa.func.count()
        else:
            # Count non null values
            return sa.func.count(x)


#### Window Functions ####


with SQLTableImpl.op(ops.Shift()) as op:

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
            return sa.func.LAG(x, by, empty_value, type_=x.type).over(
                partition_by=_window_partition_by, order_by=_window_order_by
            )
        if by < 0:
            return sa.func.LEAD(x, -by, empty_value, type_=x.type).over(
                partition_by=_window_partition_by, order_by=_window_order_by
            )


with SQLTableImpl.op(ops.RowNumber()) as op:

    @op.auto
    def _row_number():
        return sa.func.ROW_NUMBER(type_=sa.Integer())


with SQLTableImpl.op(ops.Rank()) as op:

    @op.auto
    def _rank():
        return sa.func.rank()


with SQLTableImpl.op(ops.DenseRank()) as op:

    @op.auto
    def _dense_rank():
        return sa.func.dense_rank()


from .mssql import MSSqlTableImpl  # noqa
from .duckdb import DuckDBTableImpl  # noqa
from .postgres import PostgresTableImpl  # noqa
from .sqlite import SQLiteTableImpl  # noqa
