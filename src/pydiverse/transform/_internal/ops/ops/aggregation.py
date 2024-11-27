from __future__ import annotations

from pydiverse.transform._internal.ops.op import Ftype, Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import (
    COMPARABLE,
    NUMERIC,
    Bool,
    D,
    Decimal,
    Float,
    Int,
)


class Aggregation(Operator):
    def __init__(
        self, name: str, *signatures: Signature, generate_expr_method: bool = True
    ):
        super().__init__(
            name,
            *signatures,
            ftype=Ftype.AGGREGATE,
            context_kwargs=["partition_by", "filter"],
            generate_expr_method=generate_expr_method,
        )


min = Aggregation("min", *(Signature(dtype, return_type=dtype) for dtype in COMPARABLE))

max = Aggregation("max", *(Signature(dtype, return_type=dtype) for dtype in COMPARABLE))

mean = Aggregation(
    "mean",
    *(Signature(dtype, return_type=dtype) for dtype in (Float(), Decimal())),
    Signature(Int(), return_type=Float()),
)

sum = Aggregation("sum", *(Signature(dtype, return_type=dtype) for dtype in NUMERIC))

any = Aggregation("any", Signature(Bool(), return_type=Bool()))

all = Aggregation("all", Signature(Bool(), return_type=Bool()))

count = Aggregation("count", Signature(D, return_type=Int()))

len = Aggregation("len", Signature(return_type=Int()), generate_expr_method=False)
