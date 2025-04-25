from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import COMPARABLE, Bool, S

equal = Operator(
    "__eq__", Signature(S, S, return_type=Bool()), doc="Equality comparison =="
)

not_equal = Operator(
    "__ne__", Signature(S, S, return_type=Bool()), doc="Non-equality comparison !="
)


less_than = Operator(
    "__lt__",
    *(Signature(t, t, return_type=Bool()) for t in COMPARABLE),
    doc="Less than comparison <",
)

less_equal = Operator(
    "__le__",
    *(Signature(t, t, return_type=Bool()) for t in COMPARABLE),
    doc="Less than or equal to comparison <=",
)

greater_than = Operator(
    "__gt__",
    *(Signature(t, t, return_type=Bool()) for t in COMPARABLE),
    doc="Greater than comparison >",
)

greater_equal = Operator(
    "__ge__",
    *(Signature(t, t, return_type=Bool()) for t in COMPARABLE),
    doc="Greater than or equal to comparison >=",
)

is_null = Operator(
    "is_null",
    Signature(S, return_type=Bool()),
    doc="Indicates whether the value is null.",
)

is_not_null = Operator(
    "is_not_null",
    Signature(S, return_type=Bool()),
    doc="Indicates whether the value is not null.",
)

fill_null = Operator(
    "fill_null",
    Signature(S, S, return_type=S),
    doc="Replaces every null by the given value.",
)

is_in = Operator(
    "is_in",
    Signature(S, S, ..., return_type=Bool()),
    doc="""
Whether the value equals one of the given.

Note
----
The expression ``t.c.is_in(a1, a2, ...)`` is equivalent to
``(t.c == a1) | (t.c == a2) | ...``, so passing null to ``is_in`` will result in
null. To compare for equality with null, use
:doc:`pydiverse.transform.ColExpr.is_null`.
""",
)
