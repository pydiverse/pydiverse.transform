from __future__ import annotations

from typing import Any

from pydiverse.transform._internal.ops.op import ContextKwarg, Ftype, Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import NUMERIC, Const, Int, S


class Window(Operator):
    def __init__(
        self,
        name: str,
        *signatures: Signature,
        param_names: list[str] | None = None,
        default_values: list[Any] | None = None,
        generate_expr_method=False,
        arrange_required=True,
        doc: str = "",
    ):
        super().__init__(
            name,
            *signatures,
            ftype=Ftype.WINDOW,
            context_kwargs=[
                ContextKwarg("partition_by", False),
                ContextKwarg("arrange", arrange_required),
            ],
            param_names=param_names,
            default_values=default_values,
            generate_expr_method=generate_expr_method,
            doc=doc,
        )


shift = Window(
    "shift",
    Signature(S, Const(Int()), Const(S), return_type=S),
    param_names=["self", "n", "fill_value"],
    default_values=[..., ..., None],
    generate_expr_method=True,
    arrange_required=False,
    doc="""
Shifts values in the column by an offset.

:param n:
    The number of places to shift by. May be negative.

:param fill_value:
    The value to write to the empty spaces created by the shift. Defaults to
    null.

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": [5, -1, 435, -34, 8, None, 0],
...         "b": ["r", "True", "??", ".  .", "-1/12", "abc", "#"],
...     }
... )
>>> (
...     t
...     >> mutate(
...         x=t.a.shift(2, -40),
...         y=t.b.shift(1, arrange=t.a.nulls_last()),
...     )
...     >> show()
... )
Table <unnamed>, backend: PolarsImpl
shape: (7, 4)
┌──────┬───────┬─────┬───────┐
│ a    ┆ b     ┆ x   ┆ y     │
│ ---  ┆ ---   ┆ --- ┆ ---   │
│ i64  ┆ str   ┆ i64 ┆ str   │
╞══════╪═══════╪═════╪═══════╡
│ 5    ┆ r     ┆ -40 ┆ #     │
│ -1   ┆ True  ┆ -40 ┆ .  .  │
│ 435  ┆ ??    ┆ 5   ┆ -1/12 │
│ -34  ┆ .  .  ┆ -1  ┆ null  │
│ 8    ┆ -1/12 ┆ 435 ┆ r     │
│ null ┆ abc   ┆ -34 ┆ ??    │
│ 0    ┆ #     ┆ 8   ┆ True  │
└──────┴───────┴─────┴───────┘
""",
)

row_number = Window(
    "row_number",
    Signature(return_type=Int()),
    arrange_required=False,
    doc="""
Computes the index of a row.

Via the *arrange* argument, this can be done relative to a different order of
the rows. But note that the result may not be unique if the argument of
*arrange* contains duplicates.

Examples
--------
>>> t = pdt.Table({"a": [5, -1, 435, -34, 8, None, 0]})
>>> (
...     t
...     >> mutate(
...         x=pdt.row_number(),
...         y=pdt.row_number(arrange=t.a),
...     )
...     >> show()
... )
Table <unnamed>, backend: PolarsImpl
shape: (7, 3)
┌──────┬─────┬─────┐
│ a    ┆ x   ┆ y   │
│ ---  ┆ --- ┆ --- │
│ i64  ┆ i64 ┆ i64 │
╞══════╪═════╪═════╡
│ 5    ┆ 1   ┆ 5   │
│ -1   ┆ 2   ┆ 3   │
│ 435  ┆ 3   ┆ 7   │
│ -34  ┆ 4   ┆ 2   │
│ 8    ┆ 5   ┆ 6   │
│ null ┆ 6   ┆ 1   │
│ 0    ┆ 7   ┆ 4   │
└──────┴─────┴─────┘
""",
)

rank = Window(
    "rank",
    Signature(return_type=Int()),
    doc="""
The number of strictly smaller elements in the column plus one.

This is the same as ``rank("min")`` in polars. This function has two syntax
alternatives, as shown in the example below. The pdt. version is a bit more
flexible, because it allows sorting by multiple expressions.


Examples
--------
>>> t = pdt.Table({"a": [5, -1, 435, -1, 8, None, 8]})
>>> (
...     t
...     >> mutate(
...         x=t.a.nulls_first().rank(),
...         y=pdt.rank(arrange=t.a.nulls_first()),
...     )
...     >> show()
... )
Table <unnamed>, backend: PolarsImpl
shape: (7, 3)
┌──────┬─────┬─────┐
│ a    ┆ x   ┆ y   │
│ ---  ┆ --- ┆ --- │
│ i64  ┆ i64 ┆ i64 │
╞══════╪═════╪═════╡
│ 5    ┆ 4   ┆ 4   │
│ -1   ┆ 2   ┆ 2   │
│ 435  ┆ 7   ┆ 7   │
│ -1   ┆ 2   ┆ 2   │
│ 8    ┆ 5   ┆ 5   │
│ null ┆ 1   ┆ 1   │
│ 8    ┆ 5   ┆ 5   │
└──────┴─────┴─────┘
""",
)

dense_rank = Window(
    "dense_rank",
    Signature(return_type=Int()),
    doc="""
The number of smaller or equal values in the column (not counting duplicates).

This function has two syntax alternatives, as shown in the example below. The
pdt. version is a bit more flexible, because it allows sorting by multiple
expressions.

Examples
--------
>>> t = pdt.Table({"a": [5, -1, 435, -1, 8, None, 8]})
>>> (
...     t
...     >> mutate(
...         x=t.a.nulls_first().dense_rank(),
...         y=pdt.dense_rank(arrange=t.a.nulls_first()),
...     )
...     >> show()
... )
Table <unnamed>, backend: PolarsImpl
shape: (7, 3)
┌──────┬─────┬─────┐
│ a    ┆ x   ┆ y   │
│ ---  ┆ --- ┆ --- │
│ i64  ┆ i64 ┆ i64 │
╞══════╪═════╪═════╡
│ 5    ┆ 3   ┆ 3   │
│ -1   ┆ 2   ┆ 2   │
│ 435  ┆ 5   ┆ 5   │
│ -1   ┆ 2   ┆ 2   │
│ 8    ┆ 4   ┆ 4   │
│ null ┆ 1   ┆ 1   │
│ 8    ┆ 4   ┆ 4   │
└──────┴─────┴─────┘
""",
)

prefix_sum = Window(
    "prefix_sum",
    *(Signature(dtype, return_type=dtype) for dtype in NUMERIC),
    generate_expr_method=True,
    arrange_required=False,
    doc="""
The sum of all preceding elements and the current element.
""",
)
