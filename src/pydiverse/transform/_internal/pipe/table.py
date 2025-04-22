from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from html import escape
from typing import Literal

import pandas as pd
import polars as pl
import sqlalchemy as sqa

from pydiverse.transform._internal import errors
from pydiverse.transform._internal.backend.table_impl import TableImpl
from pydiverse.transform._internal.backend.targets import Target
from pydiverse.transform._internal.errors import ColumnNotFoundError
from pydiverse.transform._internal.pipe.cache import Cache
from pydiverse.transform._internal.pipe.pipeable import Pipeable
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.col_expr import Col, ColName


class Table:
    __slots__ = ["_ast", "_cache"]

    """
    All attributes of a table are columns except for the `_ast` attribute
    which is a reference to the underlying abstract syntax tree.
    """

    def __init__(
        self,
        resource: pl.DataFrame | pl.LazyFrame | pd.DataFrame | sqa.Table | str | dict,
        backend: Target | None = None,
        *,
        name: str | None = None,
    ):
        """
        Creates a new table.

        :param resource:
            The data source to construct the table from. This can be a polars or pandas
            data frame, a python dictionary, a SQLAlchemy table or the name of a table
            in a SQL database.

        :param backend:
            The execution backend. This must be one of the pydiverse.transform backend
            objects, see :doc:`targets`. It may carry additional information how to
            interpret the *resource* argument, such as a SQLAlchemy engine.

        :param name:
            The name of the table. It is not required to give the table a name, but may
            make print output more readable.

        Examples
        --------
        **Python dictionary**.

        >>> t = pdt.Table(
        ...     {
        ...         "a": [4, 3, -35, 24, 105],
        ...         "b": [4, 4, 0, -23, 42],
        ...     },
        ...     name="T",
        ... )
        >>> t >> show()
        Table T, backend: PolarsImpl
        shape: (5, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 4   ┆ 4   │
        │ 3   ┆ 4   │
        │ -35 ┆ 0   │
        │ 24  ┆ -23 │
        │ 105 ┆ 42  │
        └─────┴─────┘

        **Polars data frame.**

        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [4, 3, -35, 24, 105],
        ...         "b": ["a", "o", "---", "i23", "  "],
        ...     },
        ... )
        >>> t = pdt.Table(df, name="T")
        >>> t >> show()
        Table T, backend: PolarsImpl
        shape: (5, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 4   ┆ a   │
        │ 3   ┆ o   │
        │ -35 ┆ --- │
        │ 24  ┆ i23 │
        │ 105 ┆     │
        └─────┴─────┘

        **Pandas data frame.** Note that the data frame is converted to a polars data
        frame and the backend is polars.

        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "a": [4, 3, -35, 24, 105],
        ...         "b": ["a", "o", "---", "i23", "  "],
        ...     },
        ... )
        >>> t = pdt.Table(df, name="T")
        >>> t >> show()
        Table T, backend: PolarsImpl
        shape: (5, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 4   ┆ a   │
        │ 3   ┆ o   │
        │ -35 ┆ --- │
        │ 24  ┆ i23 │
        │ 105 ┆     │
        └─────┴─────┘

        **SQL.** Assuming you have a SQLAlchemy engine ``engine``, which is has a
        connection to a database containing a table ``t1`` in a schema ``s1``, you can
        create a pydiverse.transform Table from it as follows.

        >>> t = pdt.Table("t1", SqlAlchemy(engine, schema="s1"))
        >>> t >> show()
        Table t1, backend: PostgresImpl
        shape: (5, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 4   ┆ a   │
        │ 3   ┆ o   │
        │ -35 ┆ --- │
        │ 24  ┆ i23 │
        │ 105 ┆     │
        └─────┴─────┘

        Note that the name argument to the ``pdt.Table`` constructor was not specified,
        so transform used the name of the SQL table. This example of course assumes that
        a database connection is set up and the above table is already present in the
        database. For more information on how to set up a connection, see
        :doc:`/database_testing`.
        """

        if not isinstance(resource, TableImpl):
            errors.check_arg_type(
                pl.DataFrame | pl.LazyFrame | pd.DataFrame | sqa.Table | str | dict,
                "Table.__init__",
                "resource",
                resource,
            )
        errors.check_arg_type(Target | None, "Table.__init__", "backend", backend)
        errors.check_arg_type(str | None, "Table.__init__", "name", name)

        self._ast: AstNode = TableImpl.from_resource(resource, backend, name=name)
        self._cache = Cache.from_ast(self._ast)

    def __getitem__(self, key: str) -> Col:
        errors.check_arg_type(str, "Table.__getitem__", "key", key)
        return self.__getattr__(key)

    def __getattr__(self, name: str) -> Col:
        if name in ("__copy__", "__deepcopy__", "__setstate__", "__getstate__"):
            # for hasattr to work correctly on dunder methods
            raise AttributeError
        if name not in self._cache.name_to_uuid:
            raise ColumnNotFoundError(
                f"column `{name}` does not exist in table `{self._ast.name}`"
            )
        col = self._cache.cols[self._cache.name_to_uuid[name]]
        return Col(name, self._ast, col._uuid, col._dtype, col._ftype)

    def __setstate__(self, d):  # to avoid very annoying AttributeErrors
        for slot, val in d[1].items():
            setattr(self, slot, val)

    def __iter__(self) -> Iterable[Col]:
        cols = self._cache.selected_cols()
        yield from cols

    def __contains__(self, col: str | Col | ColName) -> bool:
        if isinstance(col, Col):
            return col._uuid in self._cache.uuid_to_name
        if isinstance(col, ColName):
            col = col.name
        return col in self._cache.name_to_uuid

    def __len__(self) -> int:
        return len(self._cache.name_to_uuid)

    def __rshift__(self, rhs) -> Table:
        """
        The pipe operator for chaining verbs.
        """

        if isinstance(rhs, Pipeable):
            return rhs(self)
        if isinstance(rhs, Callable):
            num_params = len(inspect.signature(rhs).parameters)
            if num_params != 1:
                raise TypeError(
                    "only functions with one parameter can be used in a pipe, got "
                    f"function with {num_params} parameters."
                )
            return self >> rhs(self)

        raise TypeError(
            f"found instance of invalid type `{type(rhs)}` in the pipe. \n"
            "hint: You can use a `Table` or a Callable taking a single argument in a "
            "pipe. If you use a Callable, it will receive the current table as an "
            "and must return a `Table`."
        )

    def __str__(self):
        from pydiverse.transform._internal.backend.targets import Polars
        from pydiverse.transform._internal.pipe.verbs import export, get_backend

        backend = get_backend(self._ast)
        try:
            df = self >> export(Polars(lazy=False))
        except Exception as e:
            return (
                f"Table {self._ast.name}, backend: {backend.__name__}\n"
                f"Failed to collect table.\n{type(e).__name__}: {str(e)}"
            )

        table_str = f"Table {self._ast.name}, backend: {backend.__name__}\n{df}"
        # TODO: cache the result for a polars backend

        return table_str

    def __dir__(self) -> list[str]:
        return [name for name in self._cache.name_to_uuid.keys()]

    def _repr_html_(self) -> str | None:
        html = (
            f"Table <code>{self._ast.name}</code> using"
            f" <code>{type(self._ast).__name__}</code> backend</br>"
        )
        try:
            from pydiverse.transform._internal.backend.targets import Polars
            from pydiverse.transform._internal.pipe.verbs import export

            # TODO: For lazy backend only show preview (eg. take first 20 rows)
            # TODO: also cache the table here for a polars backend. maybe we should call
            # collect() and manage this there?
            html += (self >> export(Polars()))._repr_html_()
        except Exception as e:
            html += (
                "</br><pre>Failed to collect table.\n"
                f"{escape(e.__class__.__name__)}: {escape(str(e))}</pre>"
            )
        return html

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")


def backend(table: Table) -> Literal["polars", "sqlite", "postgres", "duckdb", "mssql"]:
    return table._cache.backend


def is_sql_backed(table: Table) -> bool:
    return table._cache.backend != "polars"
