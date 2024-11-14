from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import COMPARABLE


class Horizontal(Operator):
    def __init__(self, name: str, *signatures: Signature):
        super().__init__(
            name, *signatures, param_names=["args"], generate_expr_method=False
        )


horizontal_max = Horizontal(
    "max", *(Signature(dtype, ..., return_type=dtype) for dtype in COMPARABLE)
)

horizontal_min = Horizontal(
    "min", *(Signature(dtype, ..., return_type=dtype) for dtype in COMPARABLE)
)