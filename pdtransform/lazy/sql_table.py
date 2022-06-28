import functools
import operator
import uuid
from functools import reduce
from typing import Callable

import sqlalchemy
from sqlalchemy import sql

from pdtransform.core.column import Column
from pdtransform.core.expressions import FunctionCall, SymbolicExpression
from pdtransform.core.expressions.translator import Translator, TypedValue
from .lazy_table import JoinDescriptor, LazyTableImpl, OrderByDescriptor


class SQLTableImpl(LazyTableImpl):

    def __init__(self, engine, table):
        self.engine = sqlalchemy.create_engine(engine) if isinstance(engine, str) else engine
        tbl = self._create_table(table, self.engine)
        # backend = self.engine.url.get_backend_name()

        columns = {
            col.name: Column(name = col.name, table = self, dtype = self._convert_dtype(col.type))
            for col in tbl.columns
        }

        self.replace_tbl(tbl, columns)
        super().__init__(name = self.tbl.name, columns = columns)

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
    def _convert_dtype(dtype: sqlalchemy.sql.sqltypes.TypeEngine) -> str:
        pytype = dtype.python_type
        if pytype == int: return 'int'
        if pytype == str: return 'str'
        if pytype == bool: return 'bool'
        if pytype == float: return 'float'
        raise NotImplementedError(f"Invalid dtype {dtype}.")

    def replace_tbl(self, new_tbl, columns: dict[str: Column]):
        self.tbl = new_tbl

        self.sql_columns = {
            col.uuid: self.tbl.columns[col.name]
            for col in columns.values()
        }  # from uuid to sqlalchemy column

        if hasattr(self, 'compiled_expr'):
            # TODO: Clean up... This feels a bit hacky
            self.compiled_expr = {
                col.uuid: self.compiler.translate(col).value
                for col in columns.values()
            }

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
            self.compiled_expr[uuid](self.sql_columns).label(name)
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

    def alias(self, name):
        # TODO: If the table has not been modified, a simple `.alias()` would produce nicer queries.
        subquery = self.build_query().subquery(name=name)
        return self.__class__(self.engine, subquery)

    def collect(self):
        compiled = self.build_query()
        with self.engine.connect() as conn:
            from siuba.sql.utils import _FixedSqlDatabase
            sql_db = _FixedSqlDatabase(conn)
            return sql_db.read_sql(compiled)

    def join(self, right, on, how):
        super().join(right, on, how)

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

        descriptor = JoinDescriptor(right, on, how)
        self.joins.append(descriptor)

        self.sql_columns.update(right.sql_columns)

    def filter(self, *args):
        if self.intrinsic_grouped_by:
            self.having.extend(args)
        else:
            self.wheres.extend(args)

    def arrange(self, ordering):
        order_by = [OrderByDescriptor(col, ascending, False) for col, ascending in ordering]
        self.order_bys = order_by + self.order_bys

    def pre_summarise(self, **kwargs):
        # The result of the aggregate is always ordered according to the
        # grouping columns. We must clear the order_bys so that the order
        # is consistent with eager execution. We can do this because aggregate
        # functions are independent of the order.
        self.order_bys.clear()

        # If the grouping level is different from the grouping level of the
        # tbl object, then we must make a subquery.
        if self.intrinsic_grouped_by and self.grouped_by != self.intrinsic_grouped_by:
            # Must make a subquery
            subquery = self.build_query()
            columns = {
                name: self.get_col(name)
                for name in self.selects
            }

            self.replace_tbl(subquery, columns)

    def query_string(self):
        query = self.build_query()
        return query.compile(
            dialect = self.engine.dialect,
            compile_kwargs = {"literal_binds": True}
        )

    #### EXPRESSIONS ####

    class ExpressionCompiler(Translator['SQLTableImpl', TypedValue[Callable[[dict[uuid.UUID, sqlalchemy.Column]], sql.ColumnElement]]]):

        def _translate(self, expr, verb=None, **kwargs):
            if isinstance(expr, Column):
                # Can either be a base SQL column, or a reference to an expression
                if expr.uuid in self.backend.sql_columns:
                    def sql_col(cols):
                        return cols[expr.uuid]
                    return TypedValue(sql_col, expr.dtype)

                if expr.uuid in self.backend.compiled_expr:
                    return TypedValue(self.backend.compiled_expr[expr.uuid], self.backend.col_dtype[expr.uuid])

                raise Exception

            if isinstance(expr, FunctionCall):
                arguments = [arg.value for arg in expr.args]
                signature = tuple(arg.dtype for arg in expr.args)
                implementation = self.backend.operator_registry.get_implementation(expr.operator, signature)

                def value(cols):
                    return implementation(*(arg(cols) for arg in arguments))

                if implementation.f_type == 'a' and verb == 'mutate':
                    # Aggregate function in mutate verb -> window function
                    compiled_gb = [self.translate(group_by).value for group_by in self.backend.grouped_by]
                    def over_value(cols):
                        partition_bys = (compiled(cols) for compiled in compiled_gb)
                        return value(cols).over(
                            partition_by = sql.expression.ClauseList(*partition_bys)
                        )
                    return TypedValue(over_value, implementation.rtype)
                else:
                    return TypedValue(value, implementation.rtype)

            # Literals
            def literal_func(_):
                return expr

            if isinstance(expr, int):
                return TypedValue(literal_func, 'int')
            if isinstance(expr, float):
                return TypedValue(literal_func, 'float')
            if isinstance(expr, str):
                return TypedValue(literal_func, 'str')
            if isinstance(expr, bool):
                return TypedValue(literal_func, 'bool')

            raise NotImplementedError(expr, type(expr))


#### BACKEND SPECIFIC OPERATORS ################################################


from sqlalchemy import func as sqlfunc


@SQLTableImpl.op('__floordiv__', 'int, int -> int')
def _floordiv(x, y):
    return sql.cast(x / y, sqlalchemy.types.Integer())

@SQLTableImpl.op('__rfloordiv__', 'int, int -> int')
def _floordiv(x, y):
    return _floordiv(y, x)

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
