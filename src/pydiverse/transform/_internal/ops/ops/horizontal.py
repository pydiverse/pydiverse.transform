from __future__ import annotations

import dataclasses

from pydiverse.transform._internal.ops.op import Ftype, NoExprMethod
from pydiverse.transform._internal.ops.signature import Param, Signature
from pydiverse.transform._internal.tree.types import COMPARABLE

__all__ = ["max", "min"]


@dataclasses.dataclass(slots=True)
class Horizontal(NoExprMethod):
    ftype = Ftype.ELEMENT_WISE


max = Horizontal(
    "max",
    [Signature(Param(dtype, "args"), ..., return_type=dtype) for dtype in COMPARABLE],
)

min = Horizontal(
    "min",
    [Signature(Param(dtype, "args"), ..., return_type=dtype) for dtype in COMPARABLE],
)
