from __future__ import annotations

from typing import Any

from pydiverse.transform._internal.ops.op import ContextKwarg, Ftype, Operator
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
            context_kwargs=[
                ContextKwarg("partition_by", False),
                ContextKwarg("arrange", True),
            ],
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

rank = Window(
    "rank",
    Signature(return_type=Int()),
    doc="""
The number of strictly smaller elements in the column plus one.

This is the same as `rank("min")` in polars.

Examples
--------
>>> t = pdt.Table({"a": [3, 1, 4, 1, 5, 9, 4]})
>>> t >> mutate(b=pdt.rank(arrange=t.a)) >> export(Polars(lazy=False))
shape: (7, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ i64 │
╞═════╪═════╡
│ 3   ┆ 3   │
│ 1   ┆ 1   │
│ 4   ┆ 4   │
│ 1   ┆ 1   │
│ 5   ┆ 6   │
│ 9   ┆ 7   │
│ 4   ┆ 4   │
└─────┴─────┘
""",
)

dense_rank = Window("dense_rank", Signature(return_type=Int()))
