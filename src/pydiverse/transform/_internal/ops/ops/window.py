from __future__ import annotations

from typing import Any

from pydiverse.transform._internal.ops.op import Ftype, Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import D, Int, Tvar


class Window(Operator):
    def __init__(
        self,
        name: str,
        *signatures: Signature,
        param_names: list[str] | None = None,
        default_values: list[Any] | None = None,
        generate_expr_method=False,
        doc: str = "",
    ):
        super().__init__(
            name,
            *signatures,
            ftype=Ftype.WINDOW,
            context_kwargs=["partition_by", "arrange"],
            param_names=param_names,
            default_values=default_values,
            generate_expr_method=generate_expr_method,
            doc=doc,
        )


shift = Window(
    "shift",
    Signature(D, Int(const=True), Tvar("D", const=True), return_type=D),
    param_names=["self", "n", "fill_value"],
    default_values=[..., ..., None],
    generate_expr_method=True,
)

row_number = Window("row_number", Signature(return_type=Int()))

rank = Window("rank", Signature(return_type=Int()))

dense_rank = Window("dense_rank", Signature(return_type=Int()))
