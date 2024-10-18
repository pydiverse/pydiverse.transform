from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import COMPARABLE, D

equal = Operator("__eq__", Signature(D, return_type=D))

not_equal = Operator("__ne__", Signature(D, return_type=D))


less_than = Operator("__lt__", *(Signature(t, return_type=t) for t in COMPARABLE))

less_equal = Operator("__le__", *(Signature(t, return_type=t) for t in COMPARABLE))

greater_than = Operator("__gt__", *(Signature(t, return_type=t) for t in COMPARABLE))

greater_equal = Operator("__ge__", *(Signature(t, return_type=t) for t in COMPARABLE))

is_null = Operator("is_null", Signature(D, return_type=D))

is_not_null = Operator("is_not_null", Signature(D, return_type=D))

fill_null = Operator("fill_null", Signature(D, return_type=D))

is_in = Operator("is_in", Signature(D, D, ..., return_type=D))
