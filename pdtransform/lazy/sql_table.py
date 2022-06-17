import functools
import operator
from functools import reduce

import sqlalchemy

from pdtransform.core.column import Column
from pdtransform.core.expressions import FunctionCall
from pdtransform.core.expressions.translator import Translator, TypedValue
from .lazy_table import LazyTableImpl, JoinDescriptor, OrderByDescriptor


class SQLTableImpl(LazyTableImpl):

    def __init__(self, engine, table):
        self.engine = sqlalchemy.create_engine(engine) if isinstance(engine, str) else engine
        self.tbl = self._create_table(table, self.engine)
        self.translator = SQLExpressionTranslator(self)
        # backend = self.engine.url.get_backend_name()

        super().__init__(
            name = self.tbl.name,
            columns = {
                col.name: Column(name = col.name, table = self, dtype = self._convert_dtype(col.type))
                for col in self.tbl.columns
            }
        )

        self.sql_columns = {
            col._uuid: self.tbl.columns[col._name]
            for col in self.columns.values()
        }  # from uuid to sqlalchemy column

        self.joins = []        # type: list[JoinDescriptor]
        self.wheres = []       # type: list[SymbolicExpression]
        self.group_bys = []    # type: list[SymbolicExpression]
        self.having = []       # type: list[SymbolicExpression]
        self.order_bys = []    # type: list[OrderByDescriptor]

    def copy(self):
        c = super().copy()
        c.translator = SQLExpressionTranslator(c)
        return c

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

    def build_query(self):
        # Validate current state
        if len(self.selects) == 0:
            raise ValueError("Can't execute a SQL query without any SELECT statements.")

        # Start building query
        select = self.tbl.select()

        # FROM
        if self.joins:
            for join in self.joins:
                on = self.translator.translate(join.on).value
                select = select.join(
                    join.right.tbl,
                    onclause = on,
                    isouter = join.how != 'inner',
                    full = join.how == 'outer'
                )

        # WHERE
        if self.wheres:
            where_sexpr = functools.reduce(operator.and_, self.wheres)
            where, where_dtype = self.translator.translate(where_sexpr)
            assert(where_dtype == 'bool')
            select = select.where(where)

        # SELECT
        # Convert self.selects to SQLAlchemy Expressions
        named_cols = { name: self.col_expr[uuid] for name, uuid in self.selected_cols() }
        s = [self.translator.translate(col).value.label(name) for name, col in named_cols.items() ]
        select = select.with_only_columns(s)

        # ORDER BY
        if self.order_bys:
            o = []
            for o_by in self.order_bys:
                col = self.translator.translate(o_by.order).value
                col = col.asc() if o_by.asc else col.desc()
                col = col.nullsfirst() if o_by.nulls_first else col.nullslast()
                o.append(col)
            select = select.order_by(*o)

        return select

    #### Verb Operations ####

    def alias(self, name):
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
        self.wheres.extend(args)

    def arrange(self, ordering):
        order_by = [OrderByDescriptor(col, ascending, False) for col, ascending in ordering]
        self.order_bys = order_by + self.order_bys

    def query_string(self):
        query = self.build_query()
        return query.compile(
            dialect = self.engine.dialect,
            compile_kwargs = {"literal_binds": True}
        )


class SQLExpressionTranslator(Translator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _translate(self, expr):
        if isinstance(expr, Column):
            return TypedValue(self.backend.sql_columns[expr._uuid], expr._dtype)

        if isinstance(expr, FunctionCall):
            arguments = [arg.value for arg in expr._args]
            signature = tuple(arg.dtype for arg in expr._args)
            implementation = self.backend.operator_registry.get_implementation(expr._operator, signature)
            return TypedValue(implementation(*arguments), implementation.rtype)


        # Literals
        if isinstance(expr, int):
            return TypedValue(expr, 'int')
        if isinstance(expr, float):
            return TypedValue(expr, 'float')
        if isinstance(expr, str):
            return TypedValue(expr, 'str')
        if isinstance(expr, bool):
            return TypedValue(expr, 'bool')

        raise NotImplementedError(expr, type(expr))


from sqlalchemy import func as sqlfunc
from sqlalchemy import sql


@SQLTableImpl.op('__floordiv__', 'int, int -> int')
def _floordiv(x, y):
    return sql.cast(x / y, sqlalchemy.types.Integer())

@SQLTableImpl.op('__rfloordiv__', 'int, int -> int')
def _floordiv(x, y):
    return _floordiv(y, x)

