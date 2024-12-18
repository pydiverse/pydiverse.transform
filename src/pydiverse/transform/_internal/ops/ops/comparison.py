from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import COMPARABLE, Bool, D

equal = Operator("__eq__", Signature(D, D, return_type=Bool()))

not_equal = Operator("__ne__", Signature(D, D, return_type=Bool()))


less_than = Operator(
    "__lt__",
    *(Signature(t, t, return_type=Bool()) for t in COMPARABLE),
    doc="""
`<` as you know it.
""",
)

less_equal = Operator(
    "__le__", *(Signature(t, t, return_type=Bool()) for t in COMPARABLE)
)

greater_than = Operator(
    "__gt__", *(Signature(t, t, return_type=Bool()) for t in COMPARABLE)
)

greater_equal = Operator(
    "__ge__", *(Signature(t, t, return_type=Bool()) for t in COMPARABLE)
)

is_null = Operator("is_null", Signature(D, return_type=Bool()))

is_not_null = Operator("is_not_null", Signature(D, return_type=Bool()))

fill_null = Operator("fill_null", Signature(D, D, return_type=D))

is_in = Operator("is_in", Signature(D, D, ..., return_type=Bool()))
