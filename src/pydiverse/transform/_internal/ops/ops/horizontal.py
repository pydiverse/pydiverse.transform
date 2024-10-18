from __future__ import annotations

import dataclasses

from pydiverse.transform._internal.ops.op import Ftype, NoExprMethod
from pydiverse.transform._internal.ops.signature import Param, Signature
from pydiverse.transform._internal.tree.types import COMPARABLE


@dataclasses.dataclass(slots=True)
class Horizontal(NoExprMethod):
    ftype = Ftype.ELEMENT_WISE


horizontal_max = Horizontal(
    "max",
    *(Signature(Param(dtype, "args"), ..., return_type=dtype) for dtype in COMPARABLE),
)

horizontal_min = Horizontal(
    "min",
    *(Signature(Param(dtype, "args"), ..., return_type=dtype) for dtype in COMPARABLE),
)
