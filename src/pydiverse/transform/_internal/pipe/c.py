from __future__ import annotations

from pydiverse.transform._internal.tree.col_expr import ColName


class MC(type):
    def __getattr__(cls, name: str) -> ColName:
        return ColName(name)

    def __getitem__(cls, name: str) -> ColName:
        return ColName(name)


class C(metaclass=MC):
    pass
