from __future__ import annotations

import functools
from typing import Literal

from pydiverse.transform import tree
from pydiverse.transform.backend.table_impl import TableImpl
from pydiverse.transform.backend.targets import Target
from pydiverse.transform.core.util import (
    ordered_set,
)
from pydiverse.transform.pipe.pipeable import builtin_verb
from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree.col_expr import Col, ColExpr, ColName, Order
from pydiverse.transform.tree.verbs import (
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
def alias(expr: TableExpr, new_name: str | None = None):
    return tree.recursive_copy(expr)


@builtin_verb()
def collect(expr: TableExpr): ...


@builtin_verb()
def export(expr: TableExpr, target: Target | None = None):
    SourceBackend: type[TableImpl] = get_backend(expr)
    if target is None:
        target = SourceBackend.backend_marker()
    tree.propagate_names(expr)
    tree.propagate_types(expr)
    return SourceBackend.compile_table_expr(expr).export(target)


@builtin_verb()
def build_query(expr: TableExpr):
    return get_backend(expr).build_query(expr)


@builtin_verb()
def show_query(expr: TableExpr):
    if query := build_query(expr):
        print(query)
    else:
        print(f"No query to show for {type(expr).__name__}")

    return expr


@builtin_verb()
def select(expr: TableExpr, *args: Col | ColName):
    return Select(expr, list(args))


@builtin_verb()
def rename(expr: TableExpr, name_map: dict[str, str]):
    return Rename(expr, name_map)
    # Type check
    for k, v in name_map.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError(
                f"Key and Value of `name_map` must both be strings: ({k!r}, {v!r})"
            )

    # Reference col that doesn't exist
    if missing_cols := name_map.keys() - expr.named_cols.fwd.keys():
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
    unmodified_cols = expr.named_cols.fwd.keys() - name_map.keys()
    if duplicate_names := unmodified_cols & set(name_map.values()):
        raise ValueError(
            "Table already contains columns named: " + ", ".join(duplicate_names)
        )

    # Rename
    new_tbl = expr.copy()
    new_tbl.selects = ordered_set(name_map.get(name, name) for name in new_tbl.selects)

    uuid_name_map = {new_tbl.named_cols.fwd[old]: new for old, new in name_map.items()}
    for uuid in uuid_name_map:
        del new_tbl.named_cols.bwd[uuid]
    for uuid, name in uuid_name_map.items():
        new_tbl.named_cols.bwd[uuid] = name

    return new_tbl


@builtin_verb()
def mutate(expr: TableExpr, **kwargs: ColExpr):
    return Mutate(expr, list(kwargs.keys()), list(kwargs.values()))


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
    if suffix is None:
        suffix = f"_{right.name}"
    if suffix is None:
        suffix = "_right"
    return Join(left, right, on, how, validate, suffix)


inner_join = functools.partial(join, how="inner")
left_join = functools.partial(join, how="left")
outer_join = functools.partial(join, how="outer")


@builtin_verb()
def filter(expr: TableExpr, *args: ColExpr):
    return Filter(expr, list(args))


@builtin_verb()
def arrange(expr: TableExpr, *args: ColExpr):
    return Arrange(expr, list(Order.from_col_expr(arg) for arg in args))


@builtin_verb()
def group_by(expr: TableExpr, *args: Col | ColName, add=False):
    return GroupBy(expr, list(args), add)


@builtin_verb()
def ungroup(expr: TableExpr):
    return Ungroup(expr)


@builtin_verb()
def summarise(expr: TableExpr, **kwargs: ColExpr):
    return Summarise(expr, list(kwargs.keys()), list(kwargs.values()))


@builtin_verb()
def slice_head(expr: TableExpr, n: int, *, offset: int = 0):
    return SliceHead(expr, n, offset)


def get_backend(expr: TableExpr) -> type[TableImpl]:
    if isinstance(expr, Table):
        return expr._impl.__class__
    elif isinstance(expr, Join):
        return get_backend(expr.left)
    else:
        return get_backend(expr.table)
