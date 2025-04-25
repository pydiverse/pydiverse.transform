from __future__ import annotations

from pydiverse.transform._internal.ops.op import ContextKwarg, Ftype, Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import (
    COMPARABLE,
    NUMERIC,
    Bool,
    Const,
    Decimal,
    Float,
    Int,
    S,
    String,
)


class Aggregation(Operator):
    def __init__(
        self,
        name: str,
        *signatures: Signature,
        context_kwargs=None,
        param_names=None,
        default_values=None,
        generate_expr_method: bool = True,
        doc: str = "",
    ):
        if context_kwargs is None:
            context_kwargs = [
                ContextKwarg("partition_by", False),
                ContextKwarg("filter", False),
            ]
        super().__init__(
            name,
            *signatures,
            ftype=Ftype.AGGREGATE,
            context_kwargs=context_kwargs,
            param_names=param_names,
            default_values=default_values,
            generate_expr_method=generate_expr_method,
            doc=doc,
        )


min = Aggregation(
    "min",
    *(Signature(dtype, return_type=dtype) for dtype in COMPARABLE),
    doc="Computes the minimum value in each group.",
)

max = Aggregation(
    "max",
    *(Signature(dtype, return_type=dtype) for dtype in COMPARABLE),
    doc="Computes the maximum value in each group.",
)

mean = Aggregation(
    "mean",
    *(Signature(dtype, return_type=dtype) for dtype in (Float(), Decimal())),
    Signature(Int(), return_type=Float()),
    doc="Computes the average value in each group.",
)

sum = Aggregation(
    "sum",
    *(Signature(dtype, return_type=dtype) for dtype in NUMERIC),
    Signature(Bool(), return_type=Int()),
    doc="Computes the sum of values in each group.",
)

any = Aggregation(
    "any",
    Signature(Bool(), return_type=Bool()),
    doc="Indicates whether at least one value in a group is True.",
)

all = Aggregation(
    "all",
    Signature(Bool(), return_type=Bool()),
    doc="Indicates whether every non-null value in a group is True.",
)

count = Aggregation(
    "count",
    Signature(S, return_type=Int()),
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

str_join = Aggregation(
    "str.join",
    Signature(String(), Const(String()), return_type=String()),
    param_names=["self", "delimiter"],
    default_values=[..., ""],
    context_kwargs=[
        ContextKwarg("partition_by"),
        ContextKwarg("filter"),
        ContextKwarg("arrange"),
    ],
    doc="""
Concatenates all strings in a group to a single string.

:param delimiter:
    The string to insert between the elements.""",
)
