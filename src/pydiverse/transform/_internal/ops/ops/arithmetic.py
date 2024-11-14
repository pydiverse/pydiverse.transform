from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import (
    NUMERIC,
    Date,
    Datetime,
    Decimal,
    Duration,
    Float,
    Int,
    String,
)

add = Operator(
    "__add__",
    *(Signature(dtype, dtype, return_type=dtype) for dtype in NUMERIC),
    Signature(String(), String(), return_type=String()),
    Signature(Duration(), Duration(), return_type=Duration()),
)

sub = Operator(
    "__sub__",
    *(Signature(dtype, dtype, return_type=dtype) for dtype in NUMERIC),
    Signature(Datetime(), Datetime(), return_type=Duration()),
    Signature(Date(), Date(), return_type=Duration()),
    Signature(Datetime(), Date(), return_type=Duration()),
    Signature(Date(), Datetime(), return_type=Duration()),
)

mul = Operator(
    "__mul__", *(Signature(dtype, dtype, return_type=dtype) for dtype in NUMERIC)
)

truediv = Operator(
    "__truediv__",
    Signature(Int(), Int(), return_type=Float()),
    Signature(Float(), Float(), return_type=Float()),
    Signature(Decimal(), Decimal(), return_type=Decimal()),
)
