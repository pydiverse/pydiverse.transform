from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import S


class Marker(Operator):
    def __init__(self, name: str, doc: str = ""):
        super().__init__(
            name,
            Signature(S, return_type=S),
            doc=doc
            + """
Can only be used in expressions given to the `arrange` verb or as as an
`arrange` keyword argument.
""",
        )


nulls_first = Marker(
    "nulls_first",
    doc="""
Specifies that nulls are placed at the beginning of the ordering.

This does not mean that nulls are considered to be `less` than any other
element. I.e. if both `nulls_first` and `descending` are given, nulls will still
be placed at the beginning.

If neither `nulls_first` nor `nulls_last` is specified, the position of nulls is
backend-dependent.
""",
)

nulls_last = Marker(
    "nulls_last",
    doc="""
Specifies that nulls are placed at the end of the ordering.

This does not mean that nulls are considered to be `greater` than any other
element. I.e. if both `nulls_last` and `descending` are given, nulls will still
be placed at the end.

If neither `nulls_first` nor `nulls_last` is specified, the position of nulls is
backend-dependent.
""",
)

ascending = Marker(
    "ascending",
    doc="""
The default ordering.
""",
)

descending = Marker(
    "descending",
    doc="""
Reverses the default ordering.
""",
)
