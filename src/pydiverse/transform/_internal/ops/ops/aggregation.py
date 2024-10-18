from __future__ import annotations

from typing import Any

from pydiverse.transform._internal.ops.op import Ftype, Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import (
    COMPARABLE,
    NUMERIC,
    Bool,
    D,
    Int,
)


class Aggregation(Operator):
    def __init__(self, name: str, *signatures: Signature):
        super().__init__(
            name,
            *signatures,
            ftype=Ftype.AGGREGATE,
            context_kwargs=["partition_by", "filter"],
        )


min = Aggregation("min", *(Signature(dtype, return_type=dtype) for dtype in COMPARABLE))

max = Aggregation("max", *(Signature(dtype, return_type=dtype) for dtype in COMPARABLE))

mean = Aggregation("mean", *(Signature(dtype, return_type=dtype) for dtype in NUMERIC))

sum = Aggregation("sum", *(Signature(dtype, return_type=dtype) for dtype in NUMERIC))

any = Aggregation("any", Signature(Bool(), return_type=Bool()))

all = Aggregation("all", Signature(Bool(), return_type=Bool()))

count = Aggregation("count", Signature(D, return_type=Int()))

len = Aggregation("len", Signature(return_type=Int()))
