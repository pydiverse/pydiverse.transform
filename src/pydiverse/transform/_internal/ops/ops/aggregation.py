from __future__ import annotations

from pydiverse.transform._internal.ops.op import Aggregation
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import (
    COMPARABLE,
    NUMERIC,
    Bool,
    D,
    Int,
)

__all__ = [
    "min",
    "max",
    "mean",
    "sum",
    "any",
    "all",
    "count",
    "len",
]

min = Aggregation("min", [Signature(dtype, return_type=dtype) for dtype in COMPARABLE])

max = Aggregation("max", [Signature(dtype, return_type=dtype) for dtype in COMPARABLE])

mean = Aggregation("mean", [Signature(dtype, return_type=dtype) for dtype in NUMERIC])

sum = Aggregation("sum", [Signature(dtype, return_type=dtype) for dtype in NUMERIC])

any = Aggregation("any", [Signature(Bool(), return_type=Bool())])

all = Aggregation("all", [Signature(Bool(), return_type=Bool())])

count = Aggregation("count", [Signature(D, return_type=Int())])

len = Aggregation("len", [Signature(return_type=Int())])
