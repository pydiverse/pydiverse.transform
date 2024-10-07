from __future__ import annotations

import copy
import dataclasses
import inspect
from collections.abc import Callable, Iterable
from html import escape
from typing import Any
from uuid import UUID

from pydiverse.transform._internal.backend.table_impl import TableImpl
from pydiverse.transform._internal.backend.targets import Target
from pydiverse.transform._internal.pipe.pipeable import Pipeable
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.col_expr import Col, ColName


class Table:
    __slots__ = ["_ast", "_cache"]

    """
    All attributes of a table are columns except for the `_ast` attribute
    which is a reference to the underlying abstract syntax tree.
    """

    # TODO: define exactly what can be given for the two and do type checks
    #       maybe call the second one execution_engine or similar?
    def __init__(
        self, resource: Any, backend: Target | None = None, *, name: str | None = None
    ):
        self._ast = TableImpl.from_resource(resource, backend, name=name)
        self._cache = Cache(
            self._ast.cols,
            list(self._ast.cols.values()),
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
        return [name for name in self._cache.cols.keys() if self._cache.has_col(name)]

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
    cols: dict[str, Col]
    select: list[Col]
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
            self.all_cols = self.all_cols | {
                col._uuid: col for col in new_cols.values()
            }

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
