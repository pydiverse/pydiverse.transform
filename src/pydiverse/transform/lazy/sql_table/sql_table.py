from __future__ import annotations

import decimal
import functools
import inspect
import operator
import uuid
import warnings
from collections.abc import Iterable
from functools import reduce
from typing import Callable

import sqlalchemy as sa
from sqlalchemy import sql

from pydiverse.transform.core import ops
from pydiverse.transform.core.column import Column, LiteralColumn
from pydiverse.transform.core.expressions import SymbolicExpression, iterate_over_expr
from pydiverse.transform.core.expressions.translator import TypedValue
from pydiverse.transform.core.ops import OPType
from pydiverse.transform.core.table_impl import (
    ColumnMetaData,
    ContextInstruction,
    ImplicitArrange,
)
from pydiverse.transform.core.util import OrderingDescriptor, translate_ordering
from pydiverse.transform.lazy.lazy_table import JoinDescriptor, LazyTableImpl


class SQLTableImpl(LazyTableImpl):
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
        dataset=None,
        _dtype_hints: dict[str, str] = None,
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

        name = self.tbl.name
        if engine.dialect.name == "bigquery":
            spl = self.tbl.name.split(".")
            if len(spl) == 2:
                name = spl[1]

        super().__init__(name=name, columns=columns)

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
        if isinstance(expr, sa.sql.expression.ColumnElement):
            return str(expr.compile(compile_kwargs={"literal_binds": True}))
        return super()._html_repr_expr(expr)

    @staticmethod
    def _create_table(tbl, engine=None):
        """Return a sa.Table

        :param tbl: a sa.Table or string of form 'table_name'
            or 'schema_name.table_name'.
        """
        if isinstance(tbl, sa.sql.selectable.FromClause):
            return tbl

        if not isinstance(tbl, str):
            raise ValueError(
                "tbl must be a sqlalchemy Table or string, but was %s" % type(tbl)
            )

        spl = tbl.split(".") if "." in tbl else [None, tbl]
        schema = spl[0]
        table_name = ".".join(spl[1:])

        # TODO: pybigquery uses schema to mean project_id, so we cannot use
        #     siuba's classic breakdown "{schema}.{table_name}". Basically
        #     pybigquery uses "{schema=project_id}.{dataset_dot_table_name}" in its
        #     internal logic. An important side effect is that bigquery errors for
        #     `dataset`.`table`, but not `dataset.table`.
        if engine and engine.dialect.name == "bigquery":
            table_name = tbl
            schema = None

        return sa.Table(
            table_name,
            sa.MetaData(),
            schema=schema,
            autoload_with=engine,
        )

    @staticmethod
    def _get_dtype(col: sa.Column, hints: dict[str, str] = None) -> str:
        """Determine the dtype of a column.

        :param col: The sqlalchemy column object.
        :param hints: In some situations sqlalchemy can't determine the dtype of
            a column. Instead of throwing an exception we can use these type
            hints as a fallback.
        :return: Appropriate dtype string.
        """

        try:
            pytype = col.type.python_type
            if pytype == int:
                return "int"
            if pytype == str:
                return "str"
            if pytype == bool:
                return "bool"
            if pytype == float or pytype == decimal.Decimal:
                return "float"
            raise NotImplementedError(f"Invalid type {col.type}.")
        except NotImplementedError as e:
            if hints is not None:
                if dtype := hints.get(col.name):
                    return dtype
            raise e

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
        self.limit_offset: tuple[int, int] = None

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
        if self.joins:
            for join in self.joins:
                compiled, _ = self.compiler.translate(join.on)
                on = compiled(self.sql_columns)

                select = select.join(
                    join.right.tbl,
                    onclause=on,
                    isouter=join.how != "inner",
                    full=join.how == "outer",
                )

        # WHERE
        if self.wheres:
            # Combine wheres using ands
            combined_where = functools.reduce(
                operator.and_, map(SymbolicExpression, self.wheres)
            )._
            compiled, where_dtype = self.compiler.translate(combined_where)
            assert where_dtype == "bool"
            where = compiled(self.sql_columns)
            select = select.where(where)

        # GROUP BY
        if self.intrinsic_grouped_by:
            compiled_gb, group_by_dtypes = zip(
                *(
                    self.compiler.translate(group_by)
                    for group_by in self.intrinsic_grouped_by
                )
            )
            group_bys = (compiled(self.sql_columns) for compiled in compiled_gb)
            select = select.group_by(*group_bys)

        # HAVING
        if self.having:
            # Combine havings using ands
            combined_having = functools.reduce(
                operator.and_, map(SymbolicExpression, self.having)
            )._
            compiled, having_dtype = self.compiler.translate(combined_having)
            assert having_dtype == "bool"
            having = compiled(self.sql_columns)
            select = select.having(having)

        # LIMIT / OFFSET
        if self.limit_offset is not None:
            limit, offset = self.limit_offset
            select = select.limit(limit).offset(offset)

        # SELECT
        # Convert self.selects to SQLAlchemy Expressions
        s = []
        for name, uuid_ in self.selected_cols():
            sql_col = self.cols[uuid_].compiled(self.sql_columns)
            context = ContextInstruction()
            if isinstance(sql_col, tuple):
                sql_col, context = sql_col
            if not isinstance(sql_col, sql.ColumnElement):
                sql_col = sql.literal(sql_col)
            sql_col = context.post_process(sql_col)
            sql_col = self.post_process_select_column(sql_col, self.cols[uuid_])
            s.append(sql_col.label(name))

        select = select.with_only_columns(*s)

        # ORDER BY
        if self.order_bys:
            o = []
            for o_by in self.order_bys:
                compiled, _ = self.compiler.translate(o_by.order)
                col = compiled(self.sql_columns)
                col = self.post_process_order_by(col, o_by)
                o.append(col)
            select = select.order_by(*o)

        return select

    def get_literal_func(self, expr):
        def literal_func(*args, **kwargs):
            return sa.literal(expr)

        return literal_func

    def post_process_select_column(self, sql_col, pdt_col):
        if pdt_col.dtype == "bool":
            sql_col = sa.cast(sql_col, sa.Boolean())
        return sql_col

    def post_process_order_by(self, col, o_by):
        col = col.asc() if o_by.asc else col.desc()
        col = col.nullsfirst() if o_by.nulls_first else col.nullslast()
        return col

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
            self.order_bys.clear()

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
        try:
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
        except Exception as e:
            raise RuntimeError(
                f"Failed to execute query:\n"
                f"{select.compile(self.engine, compile_kwargs={'literal_binds': True})}"
            ) from e

        # Add metadata
        result.attrs["name"] = self.name
        return result

    def build_query(self) -> str:
        query = self.build_select()
        return str(query.compile(self.engine, compile_kwargs={"literal_binds": True}))

    def join(self, right, on, how, *, validate=None):
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
            on = reduce(operator.and_, (on, *right.wheres))
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

        if validate is not None:
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
        self.order_bys = ordering + self.order_bys

    def summarise(self, **kwargs):
        self.alignment_hash = generate_alignment_hash()

    def slice_head(self, n: int, offset: int):
        if self.limit_offset is None:
            self.limit_offset = (n, offset)
        else:
            old_n, old_o = self.limit_offset
            self.limit_offset = (min(abs(old_n - offset), n), old_o + offset)

    #### EXPRESSIONS ####

    class ExpressionCompiler(
        LazyTableImpl.ExpressionCompiler[
            "SQLTableImpl",
            TypedValue[Callable[[dict[uuid.UUID, sa.Column]], sql.ColumnElement]],
        ]
    ):
        def _get_literal_func(self, expr):
            return self.backend.get_literal_func(expr)

        def _translate_col(self, expr, **kwargs):
            # Can either be a base SQL column, or a reference to an expression
            if expr.uuid in self.backend.sql_columns:

                def sql_col(cols):
                    return cols[expr.uuid]

                return TypedValue(sql_col, expr.dtype, OPType.EWISE)

            col = self.backend.cols[expr.uuid]
            return TypedValue(col.compiled, col.dtype, col.ftype)

        def _translate_literal_col(self, expr, **kwargs):
            if not self.backend.is_aligned_with(expr):
                raise ValueError(
                    "Literal column isn't aligned with this table. "
                    f"Literal Column: {expr}"
                )

            def sql_col(cols):
                return expr.typed_value.value

            return TypedValue(sql_col, expr.typed_value.dtype, expr.typed_value.ftype)

        def _translate_function(
            self, expr, implementation, op_args, context_kwargs, verb=None, **kwargs
        ):
            def value(cols):
                args = [
                    arg.value(cols) if not is_const else arg.const_value(cols)
                    for arg, is_const in zip(op_args, implementation.const_args)
                ]
                main_args = [
                    arg[1].process_arg(arg[0]) if isinstance(arg, tuple) else arg
                    for arg in args
                ]
                parts_args = [len(arg) if isinstance(arg, tuple) else 1 for arg in args]
                res = implementation(*main_args)
                # aggregate context instructions returned by arg.value() functions
                assert (
                    len(parts_args) == 0 or max(parts_args) <= 2
                ), "value function returned more than two parts"
                return res

            operator = implementation.operator

            if operator.ftype == OPType.AGGREGATE and verb == "mutate":
                # Aggregate function in mutate verb -> window function
                over_value = self.over_clause(
                    value, implementation.operator, context_kwargs
                )
                ftype = self.backend._get_op_ftype(
                    op_args, operator, OPType.WINDOW, strict=True
                )
                return TypedValue(over_value, implementation.rtype, ftype)

            elif operator.ftype == OPType.WINDOW:
                if verb != "mutate":
                    raise ValueError(
                        "Window function are only allowed inside a mutate."
                    )

                over_value = self.over_clause(
                    value, implementation.operator, context_kwargs
                )
                ftype = self.backend._get_op_ftype(op_args, operator, strict=True)
                return TypedValue(over_value, implementation.rtype, ftype)

            else:
                ftype = self.backend._get_op_ftype(op_args, operator, strict=True)
                return TypedValue(value, implementation.rtype, ftype)

        def over_clause(
            self, value: Callable, operator: ops.Operator, context_kwargs: dict
        ):
            if operator.ftype not in (OPType.AGGREGATE, OPType.WINDOW):
                raise ValueError

            wants_order_by = operator.ftype == OPType.WINDOW

            # PARTITION BY
            compiled_pb = tuple(
                self.translate(group_by).value for group_by in self.backend.grouped_by
            )

            # ORDER BY
            def order_by_clause_generator(ordering: OrderingDescriptor):
                compiled, _ = self.translate(ordering.order)

                def clause(*args, **kwargs):
                    col = compiled(*args, **kwargs)
                    col = col.asc() if ordering.asc else col.desc()
                    if ordering.nulls_first is not None:
                        col = (
                            col.nullsfirst()
                            if ordering.nulls_first
                            else col.nullslast()
                        )
                    return col

                return clause

            # New value callable
            def over_value(*args, **kwargs):
                v = value(*args, **kwargs)
                context_instruction = ContextInstruction()
                if isinstance(v, tuple):
                    v, context_instruction = v

                pb = sql.expression.ClauseList(
                    *(compiled(*args, **kwargs) for compiled in compiled_pb)
                )
                ob = None

                if wants_order_by:
                    arrange = context_kwargs.get("arrange", [])

                    ordering = translate_ordering(self.backend, arrange)
                    compiled_ob = [order_by_clause_generator(o_by) for o_by in ordering]

                    ob = sql.expression.ClauseList(
                        *(compiled(*args, **kwargs) for compiled in compiled_ob)
                    )

                    ob = context_instruction.process_order(ob)
                    if len(ob) == 0:
                        raise TypeError("Missing 'arrange' argument.")

                res = v.over(
                    partition_by=pb,
                    order_by=ob,
                )
                res = context_instruction.post_process(res)
                return res

            return over_value

    class AlignedExpressionEvaluator(
        LazyTableImpl.AlignedExpressionEvaluator[TypedValue[sql.ColumnElement]]
    ):
        def translate(self, expr, check_alignment=True, **kwargs):
            if check_alignment:
                alignment_hashes = {
                    col.table.alignment_hash
                    for col in iterate_over_expr(expr, expand_literal_col=True)
                    if isinstance(col, Column)
                }
                if len(alignment_hashes) >= 2:
                    raise ValueError(
                        "Expression contains columns from different tables that aren't"
                        " aligned."
                    )

            return super().translate(expr, check_alignment=check_alignment, **kwargs)

        def _translate_col(self, expr, **kwargs):
            backend = expr.table
            if expr.uuid in backend.sql_columns:
                sql_col = backend.sql_columns[expr.uuid]
                return TypedValue(sql_col, expr.dtype)

            col = backend.cols[expr.uuid]
            return TypedValue(col.compiled(backend.sql_columns), col.dtype, col.ftype)

        def _translate_literal_col(self, expr, **kwargs):
            assert issubclass(expr.backend, SQLTableImpl)
            return expr.typed_value

        def _translate_function(
            self, expr, implementation, op_args, context_kwargs, **kwargs
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


def generate_alignment_hash():
    # It should be possible to have an alternative hash value that
    # is a bit more lenient -> If the same set of operations get applied
    # to a table in two different orders that produce the same table
    # object, their hash could also be equal.
    return uuid.uuid1()


# Load SQLTableImpl dialect specific subclasses
from . import dialects  # noqa


#### BACKEND SPECIFIC OPERATORS ################################################


with SQLTableImpl.op(ops.FloorDiv(), check_super=False) as op:

    @op.auto
    def _floordiv(lhs, rhs):
        return sa.cast(lhs / rhs, sa.Integer())


with SQLTableImpl.op(ops.RFloorDiv(), check_super=False) as op:

    @op.auto
    def _rfloordiv(rhs, lhs):
        return _floordiv(lhs, rhs)


with SQLTableImpl.op(ops.Abs()) as op:

    @op.auto
    def _abs(x):
        return sa.func.ABS(x)


with SQLTableImpl.op(ops.Round()) as op:

    @op.auto
    def _round(x, decimals=0):
        # TODO: Don't round integers with decimals >= 0
        if decimals >= 0:
            return sa.func.round(x, decimals)
        # For some reason SQLite doesn't like negative decimals values
        return sa.func.round(x / (10**-decimals)) * (10**-decimals)


with SQLTableImpl.op(ops.Strip()) as op:

    @op.auto
    def _strip(x):
        return sa.func.TRIM(x)


#### Summarising Functions ####


with SQLTableImpl.op(ops.Mean()) as op:

    @op.auto
    def _mean(x):
        return sa.func.AVG(x)


with SQLTableImpl.op(ops.Min()) as op:

    @op.auto
    def _min(x):
        return sa.func.MIN(x)


with SQLTableImpl.op(ops.Max()) as op:

    @op.auto
    def _max(x):
        return sa.func.MAX(x)


with SQLTableImpl.op(ops.Sum()) as op:

    @op.auto
    def _sum(x):
        return sa.func.SUM(x)


with SQLTableImpl.op(ops.Any()) as op:

    @op.auto
    def _any(x):
        return sa.func.bool_or(x)


with SQLTableImpl.op(ops.All()) as op:

    @op.auto
    def _all(x):
        return sa.func.bool_and(x)


with SQLTableImpl.op(ops.StringJoin()) as op:

    @op.auto
    def _join(x, sep: str):
        return sa.func.GROUP_CONCAT(x, sep)


with SQLTableImpl.op(ops.Count()) as op:

    @op.auto
    def _count(x=None):
        if x is None:
            # Get the number of rows
            return sa.func.COUNT()
        else:
            # Count non null values
            return sa.func.COUNT(x)


#### Window Functions ####


with SQLTableImpl.op(ops.Shift()) as op:

    @op.auto
    def _shift(x, by, empty_value=None):
        if by == 0:
            return x
        if by > 0:
            return sa.func.LAG(x, by, empty_value)
        if by < 0:
            return sa.func.LEAD(x, -by, empty_value)
        raise Exception


with SQLTableImpl.op(ops.RowNumber()) as op:

    @op.auto
    def _row_number(*args):
        # support both col.row_number() and row_number(arrange=[col,...])
        if len(args) == 0:
            return sa.func.ROW_NUMBER()
        elif len(args) == 1:
            return sa.func.ROW_NUMBER(), ImplicitArrange(
                sql.expression.ClauseList(args[0])
            )
        else:
            raise AssertionError(
                "Number of arguments should already have been checked based on "
                "signatures"
            )


with SQLTableImpl.op(ops.Rank()) as op:

    @op.auto
    def _rank(x):
        # row_number() is like rank(method="first")
        # rank() is like method="min"
        return sa.func.RANK(), ImplicitArrange(sql.expression.ClauseList(x))
