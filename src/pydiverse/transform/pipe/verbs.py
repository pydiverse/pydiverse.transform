from __future__ import annotations

import functools
from typing import Literal

from pydiverse.transform import tree
from pydiverse.transform.backend.table_impl import TableImpl
from pydiverse.transform.backend.targets import Target
from pydiverse.transform.pipe.pipeable import builtin_verb
from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree.col_expr import Col, ColExpr, ColName, Order
from pydiverse.transform.tree.verbs import (
    Arrange,
    Drop,
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
    "drop",
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
    if new_name is None:
        new_name = expr.name
    new_expr, _ = expr.clone()
    new_expr.name = new_name
    return new_expr


@builtin_verb()
def collect(expr: TableExpr): ...


@builtin_verb()
def export(expr: TableExpr, target: Target):
    expr, _ = expr.clone()
    SourceBackend: type[TableImpl] = get_backend(expr)
    tree.preprocess(expr)
    return SourceBackend.export(expr, target)


@builtin_verb()
def build_query(expr: TableExpr) -> str:
    expr, _ = expr.clone()
    SourceBackend: type[TableImpl] = get_backend(expr)
    tree.preprocess(expr)
    return SourceBackend.build_query(expr)


@builtin_verb()
def show_query(expr: TableExpr):
    if query := expr >> build_query():
        print(query)
    else:
        print(f"No query to show for {type(expr).__name__}")

    return expr


@builtin_verb()
def select(expr: TableExpr, *args: Col | ColName):
    return Select(expr, list(args))


@builtin_verb()
def drop(expr: TableExpr, *args: Col | ColName):
    return Drop(expr, list(args))


@builtin_verb()
def rename(expr: TableExpr, name_map: dict[str, str]):
    if not isinstance(name_map, dict) or not name_map:
        raise TypeError("`name_map` argument to `rename` must be a nonempty dict")
    return Rename(expr, name_map)


@builtin_verb()
def mutate(expr: TableExpr, **kwargs: ColExpr):
    if not kwargs:
        raise TypeError("`mutate` requires at least one name-column-pair")
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
    if suffix is None and right.name:
        suffix = f"_{right.name}"
    if suffix is None:
        suffix = "_right"
    return Join(left, right, on, how, validate, suffix)


inner_join = functools.partial(join, how="inner")
left_join = functools.partial(join, how="left")
outer_join = functools.partial(join, how="outer")


@builtin_verb()
def filter(expr: TableExpr, predicate: ColExpr, *additional_predicates: ColExpr):
    return Filter(expr, list(predicate, *additional_predicates))


@builtin_verb()
def arrange(expr: TableExpr, by: ColExpr, *additional_by: ColExpr):
    return Arrange(expr, list(Order.from_col_expr(ord) for ord in (by, *additional_by)))


@builtin_verb()
def group_by(
    expr: TableExpr, col: Col | ColName, *additional_cols: Col | ColName, add=False
):
    return GroupBy(expr, list(col, *additional_cols), add)


@builtin_verb()
def ungroup(expr: TableExpr):
    return Ungroup(expr)


@builtin_verb()
def summarise(expr: TableExpr, **kwargs: ColExpr):
    if not kwargs:
        # if we want to include the grouping columns after summarise by default,
        # an empty summarise should be allowed
        raise TypeError("`summarise` requires at least one name-column-pair")
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
