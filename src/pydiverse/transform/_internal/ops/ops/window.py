from __future__ import annotations

from pydiverse.transform._internal.ops.op import Window
from pydiverse.transform._internal.ops.signature import Param, Signature
from pydiverse.transform._internal.tree.types import D, Int, Tvar

shift = Window(
    "shift",
    [
        Signature(
            D,
            Param(Int(const=True), "n"),
            Param(Tvar("D", const=True), "fill_value", None),
            return_type=D,
        )
    ],
)

row_number = Window("row_number", Signature(return_type=Int()))

rank = Window("rank", Signature(return_type=Int()))

dense_rank = Window("dense_rank", Signature(return_type=Int()))
