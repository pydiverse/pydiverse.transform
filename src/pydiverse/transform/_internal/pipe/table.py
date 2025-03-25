from __future__ import annotations

import copy
import dataclasses
import inspect
from collections.abc import Callable, Iterable
from html import escape
from uuid import UUID

import pandas as pd
import polars as pl
import sqlalchemy as sqa

from pydiverse.transform._internal import errors
from pydiverse.transform._internal.backend.table_impl import TableImpl
from pydiverse.transform._internal.backend.targets import Target
from pydiverse.transform._internal.pipe.pipeable import Pipeable
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.col_expr import Col, ColName
from pydiverse.transform._internal.tree.verbs import (
    Join,
    Mutate,
    Rename,
    Select,
    Summarize,
    Verb,
)


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
        self._cache = Cache(
            list(self._ast.cols.values()),
            {col.name: col._uuid for col in self._ast.cols.values()},
            {col._uuid: col.name for col in self._ast.cols.values()},
            [],
            {self._ast},
            {col._uuid: col for col in self._ast.cols.values()},
        )

    def __getitem__(self, key: str) -> Col:
        if not isinstance(key, str):
            raise TypeError(
                f"argument to __getitem__ (bracket `[]` operator) on a Table must be a "
                f"str, got {type(key)} instead."
            )
        return self.__getattr__(key)

    def __getattr__(self, name: str) -> Col:
        if name in ("__copy__", "__deepcopy__", "__setstate__", "__getstate__"):
            # for hasattr to work correctly on dunder methods
            raise AttributeError
        if name not in self._cache.name_to_uuid:
            raise ValueError(
                f"column `{name}` does not exist in table `{self._ast.name}`"
            )
        col = self._cache.all_cols[self._cache.name_to_uuid[name]]
        return Col(name, self._ast, col._uuid, col._dtype, col._ftype)

    def __setstate__(self, d):  # to avoid very annoying AttributeErrors
        for slot, val in d[1].items():
            setattr(self, slot, val)

    def __iter__(self) -> Iterable[Col]:
        cols = copy.copy(self._cache.select)
        yield from cols

    def __contains__(self, col: str | Col | ColName) -> bool:
        return self._cache.has_col(col)

    def __len__(self) -> int:
        return len(self._cache.select)

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
        return [
            name
            for name in self._cache.name_to_uuid.keys()
            if self._cache.has_col(name)
        ]

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


@dataclasses.dataclass(slots=True)
class Cache:
    # TODO: think about what sets of columns are in the respective structures and
    # write this here.
    select: list[Col]
    name_to_uuid: dict[str, UUID]
    uuid_to_name: dict[UUID, str]  # only the selected UUIDs
    partition_by: list[Col]
    # all nodes that this table is derived from (it cannot be joined with another node
    # having nonempty intersection of `derived_from`)
    derived_from: set[AstNode]
    all_cols: dict[UUID, Col]  # all columns in current scope (including unnamed ones)

    def has_col(self, col: str | Col | ColName) -> bool:
        if isinstance(col, Col):
            return col._uuid in self.uuid_to_name
        if isinstance(col, ColName):
            col = col.name
        return col in self.name_to_uuid and self.name_to_uuid[col] in self.uuid_to_name

    def update(self, vb: Verb, rcache: Cache | None = None):
        if isinstance(vb, Select):
            self.select = vb.select
            selected_uuids = set(col._uuid for col in self.select)
            selected_names = set(
                name for uid, name in self.uuid_to_name.items() if uid in selected_uuids
            )
            self.uuid_to_name = {
                uid: name
                for uid, name in self.uuid_to_name.items()
                if name in selected_names
            }

        elif isinstance(vb, Rename):
            self.name_to_uuid = {
                (new_name if (new_name := vb.name_map.get(name)) else name): uid
                for name, uid in self.name_to_uuid.items()
            }
            self.uuid_to_name = self.uuid_to_name | {
                uid: name for name, uid in self.name_to_uuid.items()
            }

        elif isinstance(vb, Mutate | Summarize):
            if isinstance(vb, Mutate):
                new_cols = {
                    name: self.all_cols[uid] for name, uid in self.name_to_uuid.items()
                }
            else:
                new_cols = {
                    self.uuid_to_name[col._uuid]: col for col in self.partition_by
                }
            new_cols |= {
                name: Col(
                    name,
                    vb,
                    uid,
                    val.dtype(),
                    val.ftype(agg_is_window=isinstance(vb, Mutate)),
                )
                for name, val, uid in zip(vb.names, vb.values, vb.uuids, strict=True)
            }

            overwritten = {
                col_name for col_name in vb.names if col_name in self.name_to_uuid
            }

            if isinstance(vb, Mutate):
                self.select = [
                    col
                    for col in self.select
                    if self.uuid_to_name[col._uuid] not in overwritten
                ]
            else:
                self.select = self.partition_by
            self.select = self.select + [new_cols[name] for name in vb.names]

            self.all_cols = self.all_cols | {
                col._uuid: col for _, col in new_cols.items()
            }
            self.name_to_uuid = {name: col._uuid for name, col in new_cols.items()}
            self.uuid_to_name = {
                uid: name
                for uid, name in self.uuid_to_name.items()
                if name not in overwritten
            } | {uid: name for uid, name in zip(vb.uuids, vb.names, strict=False)}

            if isinstance(vb, Summarize):
                self.partition_by = []

        elif isinstance(vb, Join):
            self.select = self.select + rcache.select
            self.all_cols = self.all_cols | rcache.all_cols
            self.name_to_uuid = self.name_to_uuid | {
                name + vb.suffix: uid for name, uid in rcache.name_to_uuid.items()
            }

            self.uuid_to_name = self.uuid_to_name | {
                uid: name + vb.suffix for uid, name in rcache.uuid_to_name.items()
            }

            self.derived_from = self.derived_from | rcache.derived_from

        self.derived_from = self.derived_from | {vb}
