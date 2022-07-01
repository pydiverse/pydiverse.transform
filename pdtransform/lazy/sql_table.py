import functools
import operator
import uuid
import warnings
from functools import reduce
from typing import Callable

import sqlalchemy
from sqlalchemy import sql

from pdtransform.core.column import Column, LiteralColumn
from pdtransform.core.expressions import SymbolicExpression, iterate_over_expr
from pdtransform.core.expressions.translator import TypedValue
from pdtransform.core.table_impl import ColumnMetaData
from .lazy_table import JoinDescriptor, LazyTableImpl, OrderByDescriptor


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

    def __init__(self, engine, table, _dtype_hints: dict[str, str] = None):
        self.engine = sqlalchemy.create_engine(engine) if isinstance(engine, str) else engine
        tbl = self._create_table(table, self.engine)
        # backend = self.engine.url.get_backend_name()

        columns = {
            col.name: Column(name = col.name, table = self, dtype = self._get_dtype(col, hints=_dtype_hints))
            for col in tbl.columns
        }

        self.replace_tbl(tbl, columns)
        super().__init__(name = self.tbl.name, columns = columns)

    def is_aligned_with(self, col: Column | LiteralColumn) -> bool:
        if isinstance(col, Column):
            if not isinstance(col.table, type(self)):
                return False
            return col.table.alignment_hash == self.alignment_hash

        if isinstance(col, LiteralColumn):
            return all(
                self.is_aligned_with(atom)
                for atom in iterate_over_expr(col.expr, expand_literal_col = True)
                if isinstance(atom, Column))

        raise ValueError

    @classmethod
    def _html_repr_expr(cls, expr):
        if isinstance(expr, sqlalchemy.sql.expression.ColumnElement):
            return str(expr.compile(compile_kwargs = {'literal_binds': True}))
        return super()._html_repr_expr(expr)

    @staticmethod
    def _create_table(tbl, engine = None):
        """Return a sqlalchemy.Table

        Arguments:
            tbl: a sqlalchemy.Table or string of form 'table_name' or 'schema_name.table_name'.
            source: a sqlalchemy engine, used to autoload columns.
        """
        if isinstance(tbl, sqlalchemy.sql.selectable.FromClause):
            return tbl

        if not isinstance(tbl, str):
            raise ValueError("tbl must be a sqlalchemy Table or string, but was %s" %type(tbl))

        schema, table_name = tbl.split('.') if '.' in tbl else [None, tbl]

        # TODO: pybigquery uses schema to mean project_id, so we cannot use
        #     siuba's classic breakdown "{schema}.{table_name}". Basically
        #     pybigquery uses "{schema=project_id}.{dataset_dot_table_name}" in its internal
        #     logic. An important side effect is that bigquery errors for
        #     `dataset`.`table`, but not `dataset.table`.
        if engine and engine.dialect.name == "bigquery":
            table_name = tbl
            schema = None

        return sqlalchemy.Table(
            table_name,
            sqlalchemy.MetaData(bind = engine),
            schema = schema,
            autoload_with = engine,
        )

    @staticmethod
    def _get_dtype(col: sqlalchemy.Column, hints: dict[str, str] = None) -> str:
        """Determine the dtype of a column.

        :param col: The sqlalchemy column object.
        :param hints: In some situations sqlalchemy can't determine the dtype of
            a column. Instead of throwing an exception we can use these type
            hints as a fallback.
        :return: Appropriate dtype string.
        """

        try:
            pytype = col.type.python_type
            if pytype == int: return 'int'
            if pytype == str: return 'str'
            if pytype == bool: return 'bool'
            if pytype == float: return 'float'
            raise NotImplementedError(f"Invalid type {col.type}.")
        except NotImplementedError as e:
            if hints is not None:
                if dtype := hints.get(col.name):
                    return dtype
            raise e

    def replace_tbl(self, new_tbl, columns: dict[str: Column]):
        self.tbl = new_tbl
        self.alignment_hash = generate_alignment_hash()

        self.sql_columns = {
            col.uuid: self.tbl.columns[col.name]
            for col in columns.values()
        }  # from uuid to sqlalchemy column

        if hasattr(self, 'cols'):
            # TODO: Clean up... This feels a bit hacky
            for col in columns.values():
                self.cols[col.uuid] = ColumnMetaData.from_expr(col.uuid, col, self)

        self.joins = []         # type: list[JoinDescriptor]
        self.wheres = []        # type: list[SymbolicExpression]
        self.having = []        # type: list[SymbolicExpression]
        self.order_bys = []     # type: list[OrderByDescriptor]

    def build_query(self):
        # Validate current state
        if len(self.selects) == 0:
            raise ValueError("Can't execute a SQL query without any SELECT statements.")

        # Start building query
        select = self.tbl.select()

        # FROM
        if self.joins:
            for join in self.joins:
                compiled, _ = self.compiler.translate(join.on)
                on = compiled(self.sql_columns)

                select = select.join(
                    join.right.tbl,
                    onclause = on,
                    isouter = join.how != 'inner',
                    full = join.how == 'outer'
                )

        # WHERE
        if self.wheres:
            # Combine wheres using ands
            combined_where = functools.reduce(operator.and_, map(SymbolicExpression, self.wheres))._
            compiled, where_dtype = self.compiler.translate(combined_where)
            assert(where_dtype == 'bool')
            where = compiled(self.sql_columns)
            select = select.where(where)

        # GROUP BY
        if self.intrinsic_grouped_by:
            compiled_gb, group_by_dtypes = zip(*(self.compiler.translate(group_by) for group_by in self.intrinsic_grouped_by))
            group_bys = (compiled(self.sql_columns) for compiled in compiled_gb)
            select = select.group_by(*group_bys)

        # HAVING
        if self.having:
            # Combine havings using ands
            combined_having = functools.reduce(operator.and_, map(SymbolicExpression, self.having))._
            compiled, having_dtype = self.compiler.translate(combined_having)
            assert(having_dtype == 'bool')
            having = compiled(self.sql_columns)
            select = select.having(having)

        # SELECT
        # Convert self.selects to SQLAlchemy Expressions
        s = [
            self.cols[uuid].compiled(self.sql_columns).label(name)
            for name, uuid in self.selected_cols()
        ]
        select = select.with_only_columns(s)

        # ORDER BY
        if self.order_bys:
            o = []
            for o_by in self.order_bys:
                compiled, _ = self.compiler.translate(o_by.order)
                col = compiled(self.sql_columns)
                col = col.asc() if o_by.asc else col.desc()
                col = col.nullsfirst() if o_by.nulls_first else col.nullslast()
                o.append(col)
            select = select.order_by(*o)

        return select

    #### Verb Operations ####

    def alias(self, name=None):
        if name is None:
            suffix = format(uuid.uuid1().int % 0x7FFFFFFF, 'X')
            name = f"{self.name}_{suffix}"

        # TODO: If the table has not been modified, a simple `.alias()` would produce nicer queries.
        subquery = self.build_query().subquery(name=name)
        # In some situations sqlalchemy fails to determine the datatype of a column.
        # To circumvent this, we can pass on the information we know.
        dtype_hints = { name: self.cols[self.named_cols.fwd[name]].dtype for name in self.selects }
        return self.__class__(self.engine, subquery, _dtype_hints = dtype_hints)

    def collect(self):
        compiled = self.build_query()
        with self.engine.connect() as conn:
            from siuba.sql.utils import _FixedSqlDatabase
            sql_db = _FixedSqlDatabase(conn)
            return sql_db.read_sql(compiled)

    def pre_mutate(self, **kwargs):
        requires_subquery = any(
            self.cols[c.uuid].ftype == 'w'
            for v in kwargs.values()
            for c in iterate_over_expr(self.resolve_lambda_cols(v)) if isinstance(c, Column)
        )

        if requires_subquery:
            # TODO: It would be nice if this could be done without having to select all columns.
            #       As a potential challenge for the hackathon I propose a mean of even creating the subqueries lazyly.
            #       This means that we could perform some kind of query optimization before submitting the actual query.
            #       Eg: Instead of selecting all possible columns, only select those that actually get used.
            #       This also applies to the pre_summarise function.

            columns = {
                name: self.cols[uuid].as_column(name, self)
                for name, uuid in self.named_cols.fwd.items()
            }
            original_selects = self.selects
            self.selects = self.selects | columns.keys()
            subquery = self.build_query()

            self.replace_tbl(subquery, columns)
            self.selects = original_selects

    def join(self, right, on, how, *, validate=None):
        self.alignment_hash = generate_alignment_hash()

        # If right has joins already, merging them becomes extremely difficult
        # This is because the ON clauses could contain NULL checks in which case
        # the joins aren't associative anymore.
        if right.joins:
            raise ValueError("Can't automatically combine joins if the right side already contains a JOIN clause.")

        # TODO: Handle GROUP BY and SELECTS on left / right side

        # Combine the WHERE clauses
        if how == 'inner':
            # Inner Join: The WHERES can be combined
            self.wheres.extend(right.wheres)
        elif how == 'left':
            # WHERES from right must go into the ON clause
            on = reduce(operator.and_, (on, *right.wheres))
        elif how == 'outer':
            # For outer joins, the WHERE clause can't easily be merged.
            # The best solution for now is to move them into a subquery.
            if self.wheres:
                raise ValueError("Filters can't precede outer joins. Wrap the left side in a subquery to fix this.")
            if right.wheres:
                raise ValueError("Filters can't precede outer joins. Wrap the right side in a subquery to fix this.")

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
                    for col in iterate_over_expr(arg, expand_literal_col = True)
                    if isinstance(col, Column))

                if only_grouping_cols:
                    self.wheres.append(arg)
                else:
                    self.having.append(arg)
        else:
            self.wheres.extend(args)

    def arrange(self, ordering):
        self.alignment_hash = generate_alignment_hash()

        order_by = [OrderByDescriptor(col, ascending, False) for col, ascending in ordering]
        self.order_bys = order_by + self.order_bys

    def pre_summarise(self, **kwargs):
        # The result of the aggregate is always ordered according to the
        # grouping columns. We must clear the order_bys so that the order
        # is consistent with eager execution. We can do this because aggregate
        # functions are independent of the order.
        self.order_bys.clear()

        # If the grouping level is different from the grouping level of the
        # tbl object, or if on of the input columns is a window or aggregate
        # function, we must make a subquery.
        requires_subquery = (bool(self.intrinsic_grouped_by) and self.grouped_by != self.intrinsic_grouped_by) or any(
            self.cols[c.uuid].ftype in ('w', 'a')
            for v in kwargs.values()
            for c in iterate_over_expr(self.resolve_lambda_cols(v)) if isinstance(c, Column)
        )

        if requires_subquery:
            columns = {
                name: self.cols[uuid].as_column(name, self)
                for name, uuid in self.named_cols.fwd.items()
            }
            self.selects |= columns.keys()
            subquery = self.build_query()

            self.replace_tbl(subquery, columns)

    def summarise(self, **kwargs):
        self.alignment_hash = generate_alignment_hash()

    def query_string(self):
        query = self.build_query()
        return query.compile(
            dialect = self.engine.dialect,
            compile_kwargs = {"literal_binds": True}
        )

    #### EXPRESSIONS ####

    class ExpressionCompiler(LazyTableImpl.ExpressionCompiler['SQLTableImpl', TypedValue[Callable[[dict[uuid.UUID, sqlalchemy.Column]], sql.ColumnElement]]]):

        def _translate_col(self, expr, **kwargs):
            # Can either be a base SQL column, or a reference to an expression
            if expr.uuid in self.backend.sql_columns:
                def sql_col(cols):
                    return cols[expr.uuid]
                return TypedValue(sql_col, expr.dtype, 's')

            col = self.backend.cols[expr.uuid]
            return TypedValue(col.compiled, col.dtype, col.ftype)

        def _translate_literal_col(self, expr, **kwargs):
            if not self.backend.is_aligned_with(expr):
                raise ValueError(f"Literal column isn't aligned with this table. "
                                 f"Literal Column: {expr}")

            def sql_col(cols):
                return expr.typed_value.value

            return TypedValue(sql_col, expr.typed_value.dtype, expr.typed_value.ftype)

        def _translate_function(self, expr, arguments, implementation, verb=None, **kwargs):
            def value(cols):
                return implementation(*(arg(cols) for arg in arguments))

            if implementation.ftype == 'a' and verb == 'mutate':
                # Aggregate function in mutate verb -> window function
                compiled_gb = [self.translate(group_by).value for group_by in self.backend.grouped_by]

                def over_value(cols):
                    partition_bys = (compiled(cols) for compiled in compiled_gb)
                    return value(cols).over(
                        partition_by = sql.expression.ClauseList(*partition_bys)
                    )

                ftype = self.backend._get_func_ftype(expr.args, implementation, 'w', strict = True)
                return TypedValue(over_value, implementation.rtype, ftype)
            else:
                ftype = self.backend._get_func_ftype(expr.args, implementation, strict = True)
                return TypedValue(value, implementation.rtype, ftype)

    class AlignedExpressionEvaluator(LazyTableImpl.AlignedExpressionEvaluator[TypedValue[sql.ColumnElement]]):

        def translate(self, expr, check_alignment=True, **kwargs):
            if check_alignment:
                alignment_hashes = { col.table.alignment_hash for col in iterate_over_expr(expr, expand_literal_col = True) if isinstance(col, Column) }
                if len(alignment_hashes) >= 2:
                    raise ValueError("Expression contains columns from different tables that aren't aligned.")

            return super().translate(expr, check_alignment=check_alignment, **kwargs)

        def _translate_col(self, expr, **kwargs):
            backend = expr.table
            if expr.uuid in backend.sql_columns:
                sql_col = backend.sql_columns[expr.uuid]
                return TypedValue(sql_col, expr.dtype, 's')

            col = backend.cols[expr.uuid]
            return TypedValue(col.compiled(backend.sql_columns), col.dtype, col.ftype)

        def _translate_literal_col(self, expr, **kwargs):
            assert issubclass(expr.backend, SQLTableImpl)
            return expr.typed_value

        def _translate_function(self, expr, arguments, implementation, **kwargs):
            # Aggregate function -> window function
            value = implementation(*arguments)
            override_ftype = 'w' if implementation.ftype == 'a' else None
            ftype = SQLTableImpl._get_func_ftype(expr.args, implementation, override_ftype, strict = True)

            if implementation.ftype == 'a':
                value = value.over()

            return TypedValue(value, implementation.rtype, ftype)


def generate_alignment_hash():
    # It should be possible to have an alternative hash value that
    # is a bit more lenient -> If the same set of operations get applied
    # to a table in two different orders that produce the same table
    # object, their hash could also be equal.
    return uuid.uuid1()


#### BACKEND SPECIFIC OPERATORS ################################################


from sqlalchemy import func as sqlfunc


@SQLTableImpl.op('__floordiv__', 'int, int -> int')
def _floordiv(x, y):
    return sql.cast(x / y, sqlalchemy.types.Integer())

@SQLTableImpl.op('__rfloordiv__', 'int, int -> int')
def _floordiv(x, y):
    return _floordiv(y, x)

@SQLTableImpl.op('__round__', 'int -> int')
@SQLTableImpl.op('__round__', 'int, int -> int')
def _round(x, decimals=0):
    # Int is already rounded
    return x

@SQLTableImpl.op('__round__', 'float -> float')
@SQLTableImpl.op('__round__', 'float, int -> float')
def _round(x, decimals=0):
    # For some reason SQLite doesn't like negative values
    if decimals <= 0:
        return sqlfunc.round(x / (10 ** -decimals)) * (10 ** -decimals)
    return sqlfunc.round(x, decimals)

#### Summarising Functions ####

@SQLTableImpl.op('mean', 'int |> float')
@SQLTableImpl.op('mean', 'float |> float')
def _mean(x):
    return sqlfunc.avg(x)

@SQLTableImpl.op('min', 'int |> float')
@SQLTableImpl.op('min', 'float |> float')
@SQLTableImpl.op('min', 'str |> str')
def _min(x):
    return sqlfunc.min(x)

@SQLTableImpl.op('max', 'int |> float')
@SQLTableImpl.op('max', 'float |> float')
@SQLTableImpl.op('max', 'str |> str')
def _max(x):
    return sqlfunc.max(x)

@SQLTableImpl.op('sum', 'int |> float')
@SQLTableImpl.op('sum', 'float |> float')
def _sum(x):
    return sqlfunc.sum(x)

@SQLTableImpl.op('count', 'T |> int')
def _count(x):
    # TODO: Implement a count method that doesn't take an argument
    return sqlfunc.count()