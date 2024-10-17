from __future__ import annotations

import dataclasses

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import D


@dataclasses.dataclass(slots=True)
class Marker(Operator):
    ftype = None


nulls_first = Marker("nulls_first", [Signature(D, return_type=D)])

nulls_last = Marker("nulls_last", [Signature(D, return_type=D)])

ascending = Marker("ascending", [Signature(D, return_type=D)])

descending = Marker("descending", [Signature(D, return_type=D)])
