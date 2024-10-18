from __future__ import annotations

from pydiverse.transform._internal.ops.op import Window
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import D, Int, Tvar

shift = Window(
    "shift",
    Signature(D, Int(const=True), Tvar("D", const=True), return_type=D),
    param_names=["self", "n", "fill_value"],
    default_values=[..., ..., None],
)

row_number = Window("row_number", Signature(return_type=Int()))

rank = Window("rank", Signature(return_type=Int()))

dense_rank = Window("dense_rank", Signature(return_type=Int()))
