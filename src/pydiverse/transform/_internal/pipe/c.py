from __future__ import annotations

from pydiverse.transform._internal.tree.col_expr import ColName


class MC(type):
    def __getattr__(cls, name: str) -> ColName:
        return ColName(name)

    def __getitem__(cls, name: str) -> ColName:
        return ColName(name)


class C(metaclass=MC):
    """As an alternative to referencing a column via `<table>.<column name>`, you can
    use `C.<column name>` or `C[<column name>]`. Using :class:`C` is necessary if the
    column to be referenced does not live in a table stored as a python variable."""

    pass
