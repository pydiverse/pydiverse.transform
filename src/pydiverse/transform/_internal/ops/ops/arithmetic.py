from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import (
    NUMERIC,
    Bool,
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
    Signature(Bool(), Bool(), return_type=Int()),
    Signature(Duration(), Duration(), return_type=Duration()),
    Signature(Datetime(), Duration(), return_type=Datetime()),
    Signature(Duration(), Datetime(), return_type=Datetime()),
    doc="Addition +",
)

sub = Operator(
    "__sub__",
    *(Signature(dtype, dtype, return_type=dtype) for dtype in NUMERIC),
    Signature(Datetime(), Datetime(), return_type=Duration()),
    Signature(Date(), Date(), return_type=Duration()),
    doc="Subtraction -",
)

mul = Operator(
    "__mul__",
    *(Signature(dtype, dtype, return_type=dtype) for dtype in NUMERIC),
    doc="Multiplication *",
)

truediv = Operator(
    "__truediv__",
    Signature(Int(), Int(), return_type=Float()),
    Signature(Float(), Float(), return_type=Float()),
    Signature(Decimal(), Decimal(), return_type=Decimal()),
    doc="True division /",
)

floordiv = Operator(
    "__floordiv__",
    Signature(Int(), Int(), return_type=Int()),
    doc="""
Integer division //

Warning
-------
The behavior of this operator differs from polars and python. Polars and python
always round towards negative infinity, whereas pydiverse.transform always
rounds towards zero, regardless of the sign. This behavior matches the one of C,
C++ and all currently supported SQL backends.

See also
--------
__mod__

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": [65, -65, 65, -65],
...         "b": [7, 7, -7, -7],
...     }
... )
>>> t >> mutate(r=t.a // t.b) >> show()
shape: (4, 3)
┌─────┬─────┬─────┐
│ a   ┆ b   ┆ r   │
│ --- ┆ --- ┆ --- │
│ i64 ┆ i64 ┆ i64 │
╞═════╪═════╪═════╡
│ 65  ┆ 7   ┆ 9   │
│ -65 ┆ 7   ┆ -9  │
│ 65  ┆ -7  ┆ -9  │
│ -65 ┆ -7  ┆ 9   │
└─────┴─────┴─────┘
""",
)

mod = Operator(
    "__mod__",
    Signature(Int(), Int(), return_type=Int()),
    doc="""
The remainder of integer division %

Warning
-------
This operator behaves differently than in polars. There are at least two
conventions how `%` and :doc:`// <pydiverse.transform.ColExpr.__floordiv__>`
should behave  for negative inputs. We follow the one that C, C++ and all
currently supported SQL backends follow. This means that the output has the same
sign as the left hand side of the input, regardless of the right hand side.

See also
--------
__floordiv__

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": [65, -65, 65, -65],
...         "b": [7, 7, -7, -7],
...     }
... )
>>> t >> mutate(r=t.a % t.b) >> show()
shape: (4, 3)
┌─────┬─────┬─────┐
│ a   ┆ b   ┆ r   │
│ --- ┆ --- ┆ --- │
│ i64 ┆ i64 ┆ i64 │
╞═════╪═════╪═════╡
│ 65  ┆ 7   ┆ 2   │
│ -65 ┆ 7   ┆ -2  │
│ 65  ┆ -7  ┆ 2   │
│ -65 ┆ -7  ┆ -2  │
└─────┴─────┴─────┘
""",
)
