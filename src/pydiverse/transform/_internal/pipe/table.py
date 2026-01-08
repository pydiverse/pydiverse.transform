# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

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

        # AstNodes are also allowed for `resource`, but we do not expose this to the
        # user.
        if not isinstance(resource, AstNode):
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

    def __getitem__(self, key: str | Col | ColName) -> Col:
        errors.check_arg_type(str | Col | ColName, "Table.__getitem__", "key", key)
        if isinstance(key, ColName):
            key = key.name
        elif isinstance(key, Col):
            # This functionality is useful to get the name of a column in a table via a
            # reference of a previous table.
            if key._uuid not in self._cache.uuid_to_name:
                raise ColumnNotFoundError(
                    f"column `{key.ast_repr()}` does not exist in table `{self._ast.short_name()}`"
                )
            return Col(
                self._cache.uuid_to_name[key._uuid],
                self._ast,
                key._uuid,
                key._dtype,
                key._ftype,
            )
        return self.__getattr__(key)

    def __getattr__(self, name: str) -> Col:
        if name in ("__copy__", "__deepcopy__", "__setstate__", "__getstate__"):
            # for hasattr to work correctly on dunder methods
            raise AttributeError
        if name not in self._cache.name_to_uuid:
            raise ColumnNotFoundError(f"column `{name}` does not exist in table `{self._ast.short_name()}`")
        col = self._cache.cols[self._cache.name_to_uuid[name]]
        return Col(name, self._ast, col._uuid, col._dtype, col._ftype)

    def __setstate__(self, d):  # to avoid very annoying AttributeErrors
        for slot, val in d[1].items():
            setattr(self, slot, val)

    def __iter__(self) -> Iterable[Col]:
        cols = [self.__getattr__(name) for name in self._cache.name_to_uuid.keys()]
        yield from cols

    def __contains__(self, col: str | Col | ColName) -> bool:
        if isinstance(col, Col):
            return col._uuid in self._cache.uuid_to_name
        if isinstance(col, ColName):
            col = col.name
        return col in self._cache.name_to_uuid

    def __len__(self) -> int:
        return len(self._cache.name_to_uuid)

    def __rshift__(self, rhs) -> "Table":
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

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        from pydiverse.transform._internal.pipe.verbs import (
            build_query,
            name,
        )

        if self >> name() is None:
            tbl_name = "Table without name "
        else:
            tbl_name = f"Table `{self >> name()}` "

        res = tbl_name + f"(backend: {self._cache.backend.backend_name})\n"

        try:
            ast = repr(self._ast)
        except Exception:
            ast = ""

        try:
            query = self >> build_query()
            if query is not None:
                query = "\n\nQuery:\n" + query
            else:
                query = ""
        except Exception as e:
            return res + f"building query failed\n{type(e).__name__}: {str(e)}\n{ast}"

        df_str = ""
        if query == "":
            # consider making it configurable to show both query and df_str for sql backends
            try:
                df, height = get_head_tail(self)
                res += f"shape: ({height}, {len(self)})\n"
                df_str = str(df).split("\n", 1)[1]
            except Exception as e:
                return res + f"export failed\n{type(e).__name__}: {str(e)}\n{ast}"
        return res + df_str + query

    def __dir__(self) -> list[str]:
        return [name for name in self._cache.name_to_uuid.keys()]

    def _repr_html_(self) -> str:
        from pydiverse.transform._internal.pipe.verbs import name

        if self >> name() is None:
            tbl_name = "Table without name "
        else:
            tbl_name = f"Table <code>{self >> name()}</code> "

        html = tbl_name + f"(backend: <code>{self._cache.backend.backend_name}</code>)</br>"
        try:
            # We use polars' _repr_html_ on the first and last few rows of the table and
            # fix the `shape` afterwards.
            df, height = get_head_tail(self)
            html += f"<small>shape: ({height}, {len(self)})</small>"

            df_html = df._repr_html_()
            num_rows_begin = df_html.find("shape: (")
            num_rows_end = df_html.find(",", num_rows_begin)

        except Exception as e:
            return html + (f"</br><pre>export failed\n{escape(e.__class__.__name__)}: {escape(str(e))}</pre>")

        return f"{df_html[: num_rows_begin + 8]}{height}{df_html[num_rows_end:]}"

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")


def get_head_tail(tbl: Table) -> tuple[pl.DataFrame, int]:
    import pydiverse.transform as pdt
    from pydiverse.transform._internal.pipe.verbs import export, slice_head, summarize

    height = tbl >> summarize(num_rows=pdt.count()) >> export(pdt.Scalar)
    tbl_rows = int(pl.Config.state().get("POLARS_FMT_MAX_ROWS") or 10)
    if height <= tbl_rows:
        return tbl >> export(pdt.Polars), height
    head_tail_len = tbl_rows // 2

    # Only export the first and last few rows.
    head: pl.DataFrame = tbl >> slice_head(head_tail_len) >> export(pdt.Polars)
    tail: pl.DataFrame = (
        tbl >> slice_head(head_tail_len, offset=max(head_tail_len, height - head_tail_len)) >> export(pdt.Polars)
    )
    return pl.concat([head, tail]), height


def backend(
    table: Table,
) -> Literal["polars", "duckdb_polars", "sqlite", "postgres", "duckdb", "mssql", "ibm_db2"]:
    """
    Returns the backend of the table as a string.
    """
    return table._cache.backend.backend_name


def is_sql_backed(table: Table) -> bool:
    """
    Whether the table has a SQL backend.
    """
    return not backend(table).endswith("polars")
