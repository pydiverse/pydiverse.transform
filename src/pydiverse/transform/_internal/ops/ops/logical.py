from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import Bool

bool_and = Operator(
    "__and__",
    Signature(Bool(), Bool(), return_type=Bool()),
    doc="""
Boolean AND (__and__)

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": [True, True, True, False, False, None],
...         "b": [True, False, None, False, None, None],
...     },
...     name="bool table",
... )
>>> t >> mutate(x=t.a & t.b) >> show()
Table bool table, backend: PolarsImpl
shape: (6, 3)
┌───────┬───────┬───────┐
│ a     ┆ b     ┆ x     │
│ ---   ┆ ---   ┆ ---   │
│ bool  ┆ bool  ┆ bool  │
╞═══════╪═══════╪═══════╡
│ true  ┆ true  ┆ true  │
│ true  ┆ false ┆ false │
│ true  ┆ null  ┆ null  │
│ false ┆ false ┆ false │
│ false ┆ null  ┆ false │
│ null  ┆ null  ┆ null  │
└───────┴───────┴───────┘
""",
)

bool_or = Operator(
    "__or__",
    Signature(Bool(), Bool(), return_type=Bool()),
    doc="""
Boolean OR (__or__)

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": [True, True, True, False, False, None],
...         "b": [True, False, None, False, None, None],
...     },
...     name="bool table",
... )
>>> t >> mutate(x=t.a | t.b) >> show()
Table bool table, backend: PolarsImpl
shape: (6, 3)
┌───────┬───────┬───────┐
│ a     ┆ b     ┆ x     │
│ ---   ┆ ---   ┆ ---   │
│ bool  ┆ bool  ┆ bool  │
╞═══════╪═══════╪═══════╡
│ true  ┆ true  ┆ true  │
│ true  ┆ false ┆ true  │
│ true  ┆ null  ┆ true  │
│ false ┆ false ┆ false │
│ false ┆ null  ┆ null  │
│ null  ┆ null  ┆ null  │
└───────┴───────┴───────┘
""",
)

bool_xor = Operator(
    "__xor__",
    Signature(Bool(), Bool(), return_type=Bool()),
    doc="""
Boolean XOR (__xor__)

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": [True, True, True, False, False, None],
...         "b": [True, False, None, False, None, None],
...     },
...     name="bool table",
... )
>>> t >> mutate(x=t.a ^ t.b) >> show()
Table bool table, backend: PolarsImpl
shape: (6, 3)
┌───────┬───────┬───────┐
│ a     ┆ b     ┆ x     │
│ ---   ┆ ---   ┆ ---   │
│ bool  ┆ bool  ┆ bool  │
╞═══════╪═══════╪═══════╡
│ true  ┆ true  ┆ false │
│ true  ┆ false ┆ true  │
│ true  ┆ null  ┆ null  │
│ false ┆ false ┆ false │
│ false ┆ null  ┆ null  │
│ null  ┆ null  ┆ null  │
└───────┴───────┴───────┘
""",
)

bool_invert = Operator(
    "__invert__",
    Signature(Bool(), return_type=Bool()),
    doc="""
Boolean inversion (__invert__)

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": [True, True, True, False, False, None],
...         "b": [True, False, None, False, None, None],
...     },
...     name="bool table",
... )
>>> t >> mutate(x=~t.a) >> show()
Table bool table, backend: PolarsImpl
shape: (6, 3)
┌───────┬───────┬───────┐
│ a     ┆ b     ┆ x     │
│ ---   ┆ ---   ┆ ---   │
│ bool  ┆ bool  ┆ bool  │
╞═══════╪═══════╪═══════╡
│ true  ┆ true  ┆ false │
│ true  ┆ false ┆ false │
│ true  ┆ null  ┆ false │
│ false ┆ false ┆ true  │
│ false ┆ null  ┆ true  │
│ null  ┆ null  ┆ null  │
└───────┴───────┴───────┘
""",
)
