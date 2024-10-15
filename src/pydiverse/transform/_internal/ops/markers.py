from __future__ import annotations

from pydiverse.transform._internal.ops.core import Unary

__all__ = ["NullsFirst", "NullsLast", "Ascending", "Descending"]


class Marker(Unary):
    ftype = None


class NullsFirst(Marker):
    name = "nulls_first"
    signatures = ["T -> T"]


class NullsLast(Marker):
    name = "nulls_last"
    signatures = ["T -> T"]


class Ascending(Marker):
    name = "ascending"
    signatures = ["T -> T"]


class Descending(Marker):
    name = "descending"
    signatures = ["T -> T"]
