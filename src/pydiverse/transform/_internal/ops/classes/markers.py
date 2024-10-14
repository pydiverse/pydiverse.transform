from __future__ import annotations

from pydiverse.transform._internal.ops.operator import Operator
from pydiverse.transform._internal.ops.signature import Signature, T

__all__ = ["NullsFirst", "NullsLast", "Ascending", "Descending"]


class Marker(Operator):
    ftype = None
    signatures = [Signature(T, returns=T)]


class NullsFirst(Marker):
    name = "nulls_first"


class NullsLast(Marker):
    name = "nulls_last"


class Ascending(Marker):
    name = "ascending"


class Descending(Marker):
    name = "descending"
