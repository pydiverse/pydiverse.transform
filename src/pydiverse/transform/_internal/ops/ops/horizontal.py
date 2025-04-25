from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import (
    COMPARABLE,
    NUMERIC,
    Bool,
    Duration,
    S,
    String,
)


class Horizontal(Operator):
    def __init__(self, name: str, *signatures: Signature, doc: str = ""):
        super().__init__(
            name,
            *signatures,
            param_names=["arg", "args"],
            generate_expr_method=False,
            doc=doc,
        )


horizontal_max = Horizontal(
    "max",
    *(Signature(dtype, dtype, ..., return_type=dtype) for dtype in COMPARABLE),
    doc="""
The maximum of the given columns.

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": [5, None, 435, -1, 8, None],
...         "b": [-45, None, 6, 23, -1, 0],
...         "c": [10, None, 2, None, -53, 3],
...     }
... )
>>> t >> mutate(x=pdt.max(t.a, t.b, t.c)) >> show()
Table <unnamed>, backend: PolarsImpl
shape: (6, 4)
┌──────┬──────┬──────┬──────┐
│ a    ┆ b    ┆ c    ┆ x    │
│ ---  ┆ ---  ┆ ---  ┆ ---  │
│ i64  ┆ i64  ┆ i64  ┆ i64  │
╞══════╪══════╪══════╪══════╡
│ 5    ┆ -45  ┆ 10   ┆ 10   │
│ null ┆ null ┆ null ┆ null │
│ 435  ┆ 6    ┆ 2    ┆ 435  │
│ -1   ┆ 23   ┆ null ┆ 23   │
│ 8    ┆ -1   ┆ -53  ┆ 8    │
│ null ┆ 0    ┆ 3    ┆ 3    │
└──────┴──────┴──────┴──────┘
""",
)

horizontal_min = Horizontal(
    "min",
    *(Signature(dtype, dtype, ..., return_type=dtype) for dtype in COMPARABLE),
    doc="""
The minimum of the given columns.

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": [5, None, 435, -1, 8, None],
...         "b": [-45, None, 6, 23, -1, 0],
...         "c": [10, None, 2, None, -53, 3],
...     }
... )
>>> t >> mutate(x=pdt.min(t.a, t.b, t.c)) >> show()
Table <unnamed>, backend: PolarsImpl
shape: (6, 4)
┌──────┬──────┬──────┬──────┐
│ a    ┆ b    ┆ c    ┆ x    │
│ ---  ┆ ---  ┆ ---  ┆ ---  │
│ i64  ┆ i64  ┆ i64  ┆ i64  │
╞══════╪══════╪══════╪══════╡
│ 5    ┆ -45  ┆ 10   ┆ -45  │
│ null ┆ null ┆ null ┆ null │
│ 435  ┆ 6    ┆ 2    ┆ 2    │
│ -1   ┆ 23   ┆ null ┆ -1   │
│ 8    ┆ -1   ┆ -53  ┆ -53  │
│ null ┆ 0    ┆ 3    ┆ 0    │
└──────┴──────┴──────┴──────┘
""",
)

coalesce = Horizontal(
    "coalesce",
    Signature(S, S, ..., return_type=S),
    doc="""
Returns the first non-null value among the given.

:param arg:
    The first value.

:param args:
    Further values. All must have the same type.

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": [5, None, 435, -1, 8, None],
...         "b": [-45, None, 6, 23, 1, 0],
...         "c": [10, 2, None, None, None, None],
...     }
... )
>>> (
...     t
...     >> mutate(
...         x=pdt.coalesce(t.a, t.b, t.c),
...         y=pdt.coalesce(t.c, t.b, t.a),
...     )
...     >> show()
... )
Table <unnamed>, backend: PolarsImpl
shape: (6, 5)
┌──────┬──────┬──────┬─────┬─────┐
│ a    ┆ b    ┆ c    ┆ x   ┆ y   │
│ ---  ┆ ---  ┆ ---  ┆ --- ┆ --- │
│ i64  ┆ i64  ┆ i64  ┆ i64 ┆ i64 │
╞══════╪══════╪══════╪═════╪═════╡
│ 5    ┆ -45  ┆ 10   ┆ 5   ┆ 10  │
│ null ┆ null ┆ 2    ┆ 2   ┆ 2   │
│ 435  ┆ 6    ┆ null ┆ 435 ┆ 6   │
│ -1   ┆ 23   ┆ null ┆ -1  ┆ 23  │
│ 8    ┆ 1    ┆ null ┆ 8   ┆ 1   │
│ null ┆ 0    ┆ null ┆ 0   ┆ 0   │
└──────┴──────┴──────┴─────┴─────┘
""",
)

horizontal_any = Horizontal("any", Signature(Bool(), Bool(), ..., return_type=Bool()))

horizontal_all = Horizontal("all", Signature(Bool(), Bool(), ..., return_type=Bool()))

horizontal_sum = Horizontal(
    "sum",
    *(Signature(dtype, dtype, ..., return_type=dtype) for dtype in NUMERIC),
    Signature(String(), String(), ..., return_type=String()),
    Signature(Duration(), Duration(), ..., return_type=Duration()),
)
