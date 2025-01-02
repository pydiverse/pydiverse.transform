from __future__ import annotations

from pydiverse.transform._internal.ops.op import ContextKwarg, Ftype, Operator
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
        self,
        name: str,
        *signatures: Signature,
        generate_expr_method: bool = True,
        doc: str = "",
    ):
        super().__init__(
            name,
            *signatures,
            ftype=Ftype.AGGREGATE,
            context_kwargs=[
                ContextKwarg("partition_by", False),
                ContextKwarg("filter", False),
            ],
            generate_expr_method=generate_expr_method,
            doc=doc,
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

count = Aggregation(
    "count",
    Signature(D, return_type=Int()),
    doc="""
Counts the number of non-null elements in the column.
""",
)

count_star = Aggregation(
    "count",
    Signature(return_type=Int()),
    generate_expr_method=False,
    doc="""
Returns the number of rows of the current table, like :code:`COUNT(*)` in SQL.
""",
)
