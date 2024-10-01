from __future__ import annotations

import copy
import dataclasses
import inspect
from collections.abc import Callable, Iterable
from html import escape
from uuid import UUID

import sqlalchemy as sqa

from pydiverse.transform.backend.table_impl import TableImpl
from pydiverse.transform.pipe.pipeable import Pipeable
from pydiverse.transform.tree.ast import AstNode
from pydiverse.transform.tree.col_expr import Col, ColName


class Table:
    __slots__ = ["_ast", "_cache"]

    """
    All attributes of a table are columns except for the `_ast` attribute
    which is a reference to the underlying abstract syntax tree.
    """

    # TODO: define exactly what can be given for the two and do type checks
    #       maybe call the second one execution_engine or similar?
    def __init__(self, resource, backend=None, *, name: str | None = None):
        import polars as pl

        from pydiverse.transform.backend import (
            PolarsImpl,
            SqlAlchemy,
            SqlImpl,
        )

        if isinstance(resource, TableImpl):
            self._ast: AstNode = resource
        elif isinstance(resource, (pl.DataFrame, pl.LazyFrame)):
            if name is None:
                # TODO: we could look whether the df has a name attr set (which is the
                # case if it was previously exported)
                name = "?"
            self._ast = PolarsImpl(name, resource)
        elif isinstance(resource, (str, sqa.Table)):
            if isinstance(backend, SqlAlchemy):
                self._ast = SqlImpl(resource, backend, name)

        if self._ast is None:
            raise AssertionError

        self._cache = Cache(
            self._ast.cols,
            list(self._ast.cols.values()),
            {col._uuid: col.name for col in self._ast.cols.values()},
            [],
        )

    def __getitem__(self, key: str) -> Col:
        if not isinstance(key, str):
            raise TypeError(
                f"argument to __getitem__ (bracket `[]` operator) on a Table must be a "
                f"str, got {type(key)} instead."
            )
        if not self._cache.has_col(key):
            raise ValueError(
                f"column `{key}` does not exist in table `{self._ast.name}`"
            )
        return self._cache.cols[key]

    def __getattr__(self, name: str) -> Col:
        if name in ("__copy__", "__deepcopy__", "__setstate__", "__getstate__"):
            # for hasattr to work correctly on dunder methods
            raise AttributeError
        if not self._cache.has_col(name):
            raise ValueError(
                f"column `{name}` does not exist in table `{self._ast.name}`"
            )
        return self._cache.cols[name]

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

    def __rshift__(self, rhs):
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
        try:
            from pydiverse.transform.backend.targets import Polars
            from pydiverse.transform.pipe.verbs import export

            return (
                f"Table: {self._ast.name}, backend: {type(self._ast).__name__}\n"
                f"{self >> export(Polars())}"
            )
        except Exception as e:
            return (
                f"Table: {self._ast.name}, backend: {type(self._ast).__name__}\n"
                "failed to collect table due to an exception. "
                f"{type(e).__name__}: {str(e)}"
            )

    def __dir__(self) -> list[str]:
        return [name for name in self._cache.cols.keys() if self._cache.has_col(name)]

    def _repr_html_(self) -> str | None:
        html = (
            f"Table <code>{self._ast.name}</code> using"
            f" <code>{type(self._ast).__name__}</code> backend:</br>"
        )
        try:
            from pydiverse.transform.backend.targets import Polars
            from pydiverse.transform.pipe.verbs import export

            # TODO: For lazy backend only show preview (eg. take first 20 rows)
            html += (self >> export(Polars()))._repr_html_()
        except Exception as e:
            html += (
                "</br><pre>Failed to collect table due to an exception:\n"
                f"{escape(e.__class__.__name__)}: {escape(str(e))}</pre>"
            )
        return html

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")


@dataclasses.dataclass(slots=True)
class Cache:
    cols: dict[str, Col]
    select: list[Col]
    uuid_to_name: dict[UUID, str]  # only the selected UUIDs
    partition_by: list[Col]

    def has_col(self, col: str | Col | ColName) -> bool:
        if isinstance(col, Col):
            return col._uuid in self.uuid_to_name
        if isinstance(col, ColName):
            col = col.name
        return col in self.cols and self.cols[col]._uuid in self.uuid_to_name

    def update(
        self,
        *,
        new_select: list[Col] | None = None,
        new_cols: dict[str, Col] | None = None,
    ):
        if new_select is not None:
            self.select = new_select
        if new_cols is not None:
            self.cols = new_cols

        if new_select is not None or new_cols is not None:
            selected_uuids = (
                self.uuid_to_name
                if new_select is None
                else set(col._uuid for col in new_select)
            )
            self.uuid_to_name = {
                col._uuid: name
                for name, col in self.cols.items()
                if col._uuid in selected_uuids
            }
