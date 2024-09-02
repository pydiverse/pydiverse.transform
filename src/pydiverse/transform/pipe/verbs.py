from __future__ import annotations

import functools
from typing import Literal

from pydiverse.transform.core.util import (
    ordered_set,
)
from pydiverse.transform.pipe.pipeable import builtin_verb
from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree.col_expr import Col, ColExpr, ColName, Order
from pydiverse.transform.tree.verbs import (
    Alias,
    Arrange,
    Filter,
    GroupBy,
    Join,
    Mutate,
    Rename,
    Select,
    SliceHead,
    Summarise,
    TableExpr,
    Ungroup,
)

__all__ = [
    "alias",
    "collect",
    "build_query",
    "show_query",
    "select",
    "rename",
    "mutate",
    "join",
    "left_join",
    "inner_join",
    "outer_join",
    "filter",
    "arrange",
    "group_by",
    "ungroup",
    "summarise",
    "slice_head",
    "export",
]


@builtin_verb()
def alias(table: TableExpr, new_name: str | None = None):
    return Alias(table, new_name)


@builtin_verb()
def collect(table: TableExpr): ...


@builtin_verb()
def export(table: TableExpr): ...


@builtin_verb()
def build_query(table: TableExpr):
    return get_backend(table).build_query()


@builtin_verb()
def show_query(table: TableExpr):
    if query := build_query(table):
        print(query)
    else:
        print(f"No query to show for {type(table).__name__}")

    return table


@builtin_verb()
def select(table: TableExpr, *args: Col | ColName):
    return Select(table, list(args))


@builtin_verb()
def rename(table: TableExpr, name_map: dict[str, str]):
    return Rename(table, name_map)
    # Type check
    for k, v in name_map.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError(
                f"Key and Value of `name_map` must both be strings: ({k!r}, {v!r})"
            )

    # Reference col that doesn't exist
    if missing_cols := name_map.keys() - table.named_cols.fwd.keys():
        raise KeyError("Table has no columns named: " + ", ".join(missing_cols))

    # Can't rename two cols to the same name
    _seen = set()
    if duplicate_names := {
        name for name in name_map.values() if name in _seen or _seen.add(name)
    }:
        raise ValueError(
            "Can't rename multiple columns to the same name: "
            + ", ".join(duplicate_names)
        )

    # Can't rename a column to one that already exists
    unmodified_cols = table.named_cols.fwd.keys() - name_map.keys()
    if duplicate_names := unmodified_cols & set(name_map.values()):
        raise ValueError(
            "Table already contains columns named: " + ", ".join(duplicate_names)
        )

    # Rename
    new_tbl = table.copy()
    new_tbl.selects = ordered_set(name_map.get(name, name) for name in new_tbl.selects)

    uuid_name_map = {new_tbl.named_cols.fwd[old]: new for old, new in name_map.items()}
    for uuid in uuid_name_map:
        del new_tbl.named_cols.bwd[uuid]
    for uuid, name in uuid_name_map.items():
        new_tbl.named_cols.bwd[uuid] = name

    return new_tbl


@builtin_verb()
def mutate(table: TableExpr, **kwargs: ColExpr):
    return Mutate(table, list(kwargs.keys()), list(kwargs.values()))


@builtin_verb()
def join(
    left: TableExpr,
    right: TableExpr,
    on: ColExpr,
    how: Literal["inner", "left", "outer"],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,  # appended to cols of the right table
):
    # TODO: col name collision resolution
    return Join(left, right, on, how, validate, suffix)


inner_join = functools.partial(join, how="inner")
left_join = functools.partial(join, how="left")
outer_join = functools.partial(join, how="outer")


@builtin_verb()
def filter(table: TableExpr, *args: ColExpr):
    return Filter(table, list(args))


@builtin_verb()
def arrange(table: TableExpr, *args: ColExpr):
    return Arrange(table, list(Order.from_col_expr(arg) for arg in args))


@builtin_verb()
def group_by(table: TableExpr, *args: Col | ColName, add=False):
    return GroupBy(table, list(args), add)


@builtin_verb()
def ungroup(table: TableExpr):
    return Ungroup(table)


@builtin_verb()
def summarise(table: TableExpr, **kwargs: ColExpr):
    return Summarise(table, list(kwargs.keys()), list(kwargs.values()))


@builtin_verb()
def slice_head(table: TableExpr, n: int, *, offset: int = 0):
    return SliceHead(table, n, offset)


def get_backend(expr: TableExpr) -> type:
    if isinstance(expr, Table):
        return expr._impl.__class__
    elif isinstance(expr, Join):
        return get_backend(expr.left)
    else:
        return get_backend(expr.table)
