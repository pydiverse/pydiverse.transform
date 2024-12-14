from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import NUMERIC, Bool, Decimal, Float, Int

mod = Operator("__mod__", Signature(Int(), Int(), return_type=Int()))


pow = Operator(
    "__pow__",
    Signature(Int(), Int(), return_type=Float()),
    Signature(Float(), Float(), return_type=Float()),
    Signature(Decimal(), Decimal(), return_type=Decimal()),
)

neg = Operator("__neg__", *(Signature(t, return_type=t) for t in NUMERIC))

pos = Operator("__pos__", *(Signature(t, return_type=t) for t in NUMERIC))

abs = Operator("abs", *(Signature(t, return_type=t) for t in NUMERIC))

round = Operator(
    "round",
    *(Signature(t, Int(const=True), return_type=t) for t in NUMERIC),
    param_names=["self", "decimals"],
    default_values=[..., 0],
)

floor = Operator(
    "floor",
    Signature(Float(), return_type=Float()),
    Signature(Decimal(), return_type=Decimal()),
)

ceil = Operator(
    "ceil",
    Signature(Float(), return_type=Float()),
    Signature(Decimal(), return_type=Decimal()),
)

log = Operator("log", Signature(Float(), return_type=Float()))

exp = Operator("exp", Signature(Float(), return_type=Float()))

is_inf = Operator("is_inf", Signature(Float(), return_type=Bool()))

is_not_inf = Operator("is_not_inf", Signature(Float(), return_type=Bool()))

is_nan = Operator("is_nan", Signature(Float(), return_type=Bool()))

is_not_nan = Operator("is_not_nan", Signature(Float(), return_type=Bool()))
