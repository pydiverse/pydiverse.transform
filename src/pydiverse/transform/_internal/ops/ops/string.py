from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree.types import (
    Bool,
    Const,
    Date,
    Datetime,
    Int,
    String,
)


class StrUnary(Operator):
    def __init__(self, name: str, doc: str = ""):
        super().__init__(name, Signature(String(), return_type=String()), doc=doc)


str_strip = StrUnary(
    "str.strip",
    doc="""
Removes leading and trailing whitespace.

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": ["  BCD ", "-- 00", " A^^u", "-O2"],
...         "b": ["12431", "transform", "12__*m", "   "],
...     },
...     name="string table",
... )
>>> t >> mutate(j=t.a.str.strip(), k=t.b.str.strip()) >> show()
Table string table, backend: PolarsImpl
shape: (4, 4)
┌────────┬───────────┬───────┬───────────┐
│ a      ┆ b         ┆ j     ┆ k         │
│ ---    ┆ ---       ┆ ---   ┆ ---       │
│ str    ┆ str       ┆ str   ┆ str       │
╞════════╪═══════════╪═══════╪═══════════╡
│   BCD  ┆ 12431     ┆ BCD   ┆ 12431     │
│ -- 00  ┆ transform ┆ -- 00 ┆ transform │
│  A^^u  ┆ 12__*m    ┆ A^^u  ┆ 12__*m    │
│ -O2    ┆           ┆ -O2   ┆           │
└────────┴───────────┴───────┴───────────┘
""",
)
str_upper = StrUnary(
    "str.upper",
    doc="""
Converts all alphabet letters to upper case.

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": ["  BCD ", "-- 00", " A^^u", "-O2"],
...         "b": ["12431", "transform", "12__*m", "   "],
...     },
...     name="string table",
... )
>>> t >> mutate(j=t.a.str.upper(), k=t.b.str.upper()) >> show()
Table string table, backend: PolarsImpl
shape: (4, 4)
┌────────┬───────────┬────────┬───────────┐
│ a      ┆ b         ┆ j      ┆ k         │
│ ---    ┆ ---       ┆ ---    ┆ ---       │
│ str    ┆ str       ┆ str    ┆ str       │
╞════════╪═══════════╪════════╪═══════════╡
│   BCD  ┆ 12431     ┆   BCD  ┆ 12431     │
│ -- 00  ┆ transform ┆ -- 00  ┆ TRANSFORM │
│  A^^u  ┆ 12__*m    ┆  A^^U  ┆ 12__*M    │
│ -O2    ┆           ┆ -O2    ┆           │
└────────┴───────────┴────────┴───────────┘
""",
)
str_lower = StrUnary(
    "str.lower",
    doc="""
Converts all alphabet letters to lower case.

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": ["  BCD ", "-- 00", " A^^u", "-O2"],
...         "b": ["12431", "transform", "12__*m", "   "],
...     },
...     name="string table",
... )
>>> t >> mutate(j=t.a.str.lower(), k=t.b.str.lower()) >> show()
Table string table, backend: PolarsImpl
shape: (4, 4)
┌────────┬───────────┬────────┬───────────┐
│ a      ┆ b         ┆ j      ┆ k         │
│ ---    ┆ ---       ┆ ---    ┆ ---       │
│ str    ┆ str       ┆ str    ┆ str       │
╞════════╪═══════════╪════════╪═══════════╡
│   BCD  ┆ 12431     ┆   bcd  ┆ 12431     │
│ -- 00  ┆ transform ┆ -- 00  ┆ transform │
│  A^^u  ┆ 12__*m    ┆  a^^u  ┆ 12__*m    │
│ -O2    ┆           ┆ -o2    ┆           │
└────────┴───────────┴────────┴───────────┘
""",
)

# We should write something about number of chars vs number of bytes here.
str_len = Operator(
    "str.len",
    Signature(String(), return_type=Int()),
    doc="""
Computes the length of the string.

Leading and trailing whitespace is included in the length.

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": ["  BCD ", "-- 00", " A^^u", "-O2"],
...         "b": ["12431", "transform", "12__*m", "   "],
...     },
...     name="string table",
... )
>>> t >> mutate(j=t.a.str.len(), k=t.b.str.len()) >> show()
Table string table, backend: PolarsImpl
shape: (4, 4)
┌────────┬───────────┬─────┬─────┐
│ a      ┆ b         ┆ j   ┆ k   │
│ ---    ┆ ---       ┆ --- ┆ --- │
│ str    ┆ str       ┆ i64 ┆ i64 │
╞════════╪═══════════╪═════╪═════╡
│   BCD  ┆ 12431     ┆ 6   ┆ 5   │
│ -- 00  ┆ transform ┆ 5   ┆ 9   │
│  A^^u  ┆ 12__*m    ┆ 5   ┆ 6   │
│ -O2    ┆           ┆ 3   ┆ 3   │
└────────┴───────────┴─────┴─────┘
""",
)

str_replace_all = Operator(
    "str.replace_all",
    Signature(String(), Const(String()), Const(String()), return_type=String()),
    param_names=["self", "substr", "replacement"],
    doc="""
Replaces all occurrences of a given substring by a different string.

:param substr:
    The string to replace.

:param replacement:
    The replacement string.

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": ["  BCD ", "-- 00", " A^^u", "-O2", ""],
...         "b": ["12431", "transform", "12__*m", "   ", "abbabbabba"],
...     },
...     name="string table",
... )
>>> (
...     t
...     >> mutate(
...         r=t.a.str.replace_all("-", "?"),
...         s=t.b.str.replace_all("ansf", "[---]"),
...         u=t.b.str.replace_all("abba", "#"),
...     )
...     >> show()
... )
Table string table, backend: PolarsImpl
shape: (5, 5)
┌────────┬────────────┬────────┬────────────┬───────────┐
│ a      ┆ b          ┆ r      ┆ s          ┆ u         │
│ ---    ┆ ---        ┆ ---    ┆ ---        ┆ ---       │
│ str    ┆ str        ┆ str    ┆ str        ┆ str       │
╞════════╪════════════╪════════╪════════════╪═══════════╡
│   BCD  ┆ 12431      ┆   BCD  ┆ 12431      ┆ 12431     │
│ -- 00  ┆ transform  ┆ ?? 00  ┆ tr[---]orm ┆ transform │
│  A^^u  ┆ 12__*m     ┆  A^^u  ┆ 12__*m     ┆ 12__*m    │
│ -O2    ┆            ┆ ?O2    ┆            ┆           │
│        ┆ abbabbabba ┆        ┆ abbabbabba ┆ #bb#      │
└────────┴────────────┴────────┴────────────┴───────────┘
""",
)

str_starts_with = Operator(
    "str.starts_with",
    Signature(String(), Const(String()), return_type=Bool()),
    param_names=["self", "prefix"],
    doc="""
Whether the string starts with a given prefix.

:param prefix:
    The prefix to check.

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": ["  BCD ", "-- 00", " A^^u", "-O2", ""],
...         "b": ["12431", "transform", "12__*m", "   ", "abbabbabba"],
...     },
...     name="string table",
... )
>>> (
...     t
...     >> mutate(
...         j=t.a.str.starts_with("-"),
...         k=t.b.str.starts_with("12"),
...     )
...     >> show()
... )
Table string table, backend: PolarsImpl
shape: (5, 4)
┌────────┬────────────┬───────┬───────┐
│ a      ┆ b          ┆ j     ┆ k     │
│ ---    ┆ ---        ┆ ---   ┆ ---   │
│ str    ┆ str        ┆ bool  ┆ bool  │
╞════════╪════════════╪═══════╪═══════╡
│   BCD  ┆ 12431      ┆ false ┆ true  │
│ -- 00  ┆ transform  ┆ true  ┆ false │
│  A^^u  ┆ 12__*m     ┆ false ┆ true  │
│ -O2    ┆            ┆ true  ┆ false │
│        ┆ abbabbabba ┆ false ┆ false │
└────────┴────────────┴───────┴───────┘
""",
)

str_ends_with = Operator(
    "str.ends_with",
    Signature(String(), Const(String()), return_type=Bool()),
    param_names=["self", "suffix"],
    doc="""
Whether the string ends with a given suffix.

:param suffix:
    The suffix to check.

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": ["  BCD ", "-- 00", " A^^u", "-O2", ""],
...         "b": ["12431", "transform", "12__*m", "   ", "abbabbabba"],
...     },
...     name="string table",
... )
>>> (
...     t
...     >> mutate(
...         j=t.a.str.ends_with(""),
...         k=t.b.str.ends_with("m"),
...         l=t.a.str.ends_with("^u"),
...     )
...     >> show()
... )
Table string table, backend: PolarsImpl
shape: (5, 5)
┌────────┬────────────┬──────┬───────┬───────┐
│ a      ┆ b          ┆ j    ┆ k     ┆ l     │
│ ---    ┆ ---        ┆ ---  ┆ ---   ┆ ---   │
│ str    ┆ str        ┆ bool ┆ bool  ┆ bool  │
╞════════╪════════════╪══════╪═══════╪═══════╡
│   BCD  ┆ 12431      ┆ true ┆ false ┆ false │
│ -- 00  ┆ transform  ┆ true ┆ true  ┆ false │
│  A^^u  ┆ 12__*m     ┆ true ┆ true  ┆ true  │
│ -O2    ┆            ┆ true ┆ false ┆ false │
│        ┆ abbabbabba ┆ true ┆ false ┆ false │
└────────┴────────────┴──────┴───────┴───────┘
""",
)


str_contains = Operator(
    "str.contains",
    Signature(String(), Const(String()), return_type=Bool()),
    param_names=["self", "substr"],
    doc="""
Whether the string contains a given substring.

:param substr:
    The substring to look for.

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": ["  BCD ", "-- 00", " A^^u", "-O2", ""],
...         "b": ["12431", "transform", "12__*m", "   ", "abbabbabba"],
...     },
...     name="string table",
... )
>>> (
...     t
...     >> mutate(
...         j=t.a.str.contains(" "),
...         k=t.b.str.contains("a"),
...         l=t.b.str.contains(""),
...     )
...     >> show()
... )
Table string table, backend: PolarsImpl
shape: (5, 5)
┌────────┬────────────┬───────┬───────┬──────┐
│ a      ┆ b          ┆ j     ┆ k     ┆ l    │
│ ---    ┆ ---        ┆ ---   ┆ ---   ┆ ---  │
│ str    ┆ str        ┆ bool  ┆ bool  ┆ bool │
╞════════╪════════════╪═══════╪═══════╪══════╡
│   BCD  ┆ 12431      ┆ true  ┆ false ┆ true │
│ -- 00  ┆ transform  ┆ true  ┆ true  ┆ true │
│  A^^u  ┆ 12__*m     ┆ true  ┆ false ┆ true │
│ -O2    ┆            ┆ false ┆ false ┆ true │
│        ┆ abbabbabba ┆ false ┆ true  ┆ true │
└────────┴────────────┴───────┴───────┴──────┘
""",
)

str_slice = Operator(
    "str.slice",
    Signature(String(), Int(), Int(), return_type=String()),
    param_names=["self", "offset", "n"],
    doc="""
Returns a substring of the input string.

:param offset:
    The 0-based index of the first character included in the result.

:param n:
    The number of characters to include. If the string is shorter than *offset*
    + *n*, the result only includes as many characters as there are.

Examples
--------
>>> t = pdt.Table(
...     {
...         "a": ["  BCD ", "-- 00", " A^^u", "-O2", ""],
...         "b": ["12431", "transform", "12__*m", "   ", "abbabbabba"],
...     },
...     name="string table",
... )
>>> (
...     t
...     >> mutate(
...         j=t.a.str.slice(0, 2),
...         k=t.b.str.slice(4, 10),
...     )
...     >> show()
... )
Table string table, backend: PolarsImpl
shape: (5, 4)
┌────────┬────────────┬─────┬────────┐
│ a      ┆ b          ┆ j   ┆ k      │
│ ---    ┆ ---        ┆ --- ┆ ---    │
│ str    ┆ str        ┆ str ┆ str    │
╞════════╪════════════╪═════╪════════╡
│   BCD  ┆ 12431      ┆     ┆ 1      │
│ -- 00  ┆ transform  ┆ --  ┆ sform  │
│  A^^u  ┆ 12__*m     ┆  A  ┆ *m     │
│ -O2    ┆            ┆ -O  ┆        │
│        ┆ abbabbabba ┆     ┆ bbabba │
└────────┴────────────┴─────┴────────┘
""",
)

str_to_datetime = Operator(
    "str.to_datetime", Signature(String(), return_type=Datetime())
)

str_to_date = Operator("str.to_date", Signature(String(), return_type=Date()))
