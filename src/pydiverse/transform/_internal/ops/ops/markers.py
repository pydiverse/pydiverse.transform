from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import D


class Marker(Operator):
    def __init__(self, name: str):
        super().__init__(self, name, Signature(D, return_type=D))


nulls_first = Marker("nulls_first")

nulls_last = Marker("nulls_last")

ascending = Marker("ascending")

descending = Marker("descending")
