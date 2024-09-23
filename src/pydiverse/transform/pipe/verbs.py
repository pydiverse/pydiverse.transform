from __future__ import annotations

import functools
from typing import Literal

from pydiverse.transform.backend.table_impl import TableImpl
from pydiverse.transform.backend.targets import Target
from pydiverse.transform.pipe.pipeable import builtin_verb
from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree import verbs
from pydiverse.transform.tree.col_expr import Col, ColExpr, ColName, Order, wrap_literal
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
    Verb,
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
    # TableExpr._clone relies on the tables in a tree to be unique (it does not keep a
    # memo like __deepcopy__)
    check_table_references(expr)
    new_expr, _ = expr._clone()
    new_expr.name = new_name
    return new_expr


@builtin_verb()
def collect(expr: TableExpr): ...


@builtin_verb()
def export(expr: TableExpr, target: Target):
    check_table_references(expr)
    expr, _ = expr._clone()
    SourceBackend: type[TableImpl] = get_backend(expr)
    return SourceBackend.export(expr, target)


@builtin_verb()
def build_query(expr: TableExpr) -> str:
    check_table_references(expr)
    expr, _ = expr._clone()
    SourceBackend: type[TableImpl] = get_backend(expr)
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
    return Mutate(expr, list(kwargs.keys()), wrap_literal(list(kwargs.values())))


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
    return Join(left, right, wrap_literal(on), how, validate, suffix)


inner_join = functools.partial(join, how="inner")
left_join = functools.partial(join, how="left")
outer_join = functools.partial(join, how="outer")


@builtin_verb()
def filter(expr: TableExpr, predicate: ColExpr, *additional_predicates: ColExpr):
    return Filter(expr, wrap_literal(list((predicate, *additional_predicates))))


@builtin_verb()
def arrange(expr: TableExpr, by: ColExpr, *additional_by: ColExpr):
    return Arrange(
        expr,
        wrap_literal(list(Order.from_col_expr(ord) for ord in (by, *additional_by))),
    )


@builtin_verb()
def group_by(
    expr: TableExpr, col: Col | ColName, *additional_cols: Col | ColName, add=False
):
    return GroupBy(expr, wrap_literal(list((col, *additional_cols))), add)


@builtin_verb()
def ungroup(expr: TableExpr):
    return Ungroup(expr)


@builtin_verb()
def summarise(expr: TableExpr, **kwargs: ColExpr):
    if not kwargs:
        # if we want to include the grouping columns after summarise by default,
        # an empty summarise should be allowed
        raise TypeError("`summarise` requires at least one name-column-pair")
    return Summarise(expr, list(kwargs.keys()), wrap_literal(list(kwargs.values())))


@builtin_verb()
def slice_head(expr: TableExpr, n: int, *, offset: int = 0):
    return SliceHead(expr, n, offset)


# checks whether there are duplicate tables and whether all cols used in expressions
# have are from descendants
def check_table_references(expr: TableExpr) -> set[TableExpr]:
    if isinstance(expr, verbs.Verb):
        tables = check_table_references(expr.table)

        if isinstance(expr, verbs.Join):
            right_tables = check_table_references(expr.right)
            if intersection := tables & right_tables:
                raise ValueError(
                    f"table `{list(intersection)[0]}` occurs twice in the table "
                    "tree.\nhint: To join two tables derived from a common table, "
                    "apply `>> alias()` to one of them before the join."
                )

            if len(right_tables) > len(tables):
                tables, right_tables = right_tables, tables
            tables |= right_tables

        for col in expr._iter_col_nodes():
            if isinstance(col, Col) and col.table not in tables:
                raise ValueError(
                    f"table `{col.table}` referenced via column `{col}` cannot be "
                    "used at this point. It The current table is not derived "
                    "from it."
                )

        tables.add(expr)
        return tables

    else:
        return {expr}


def get_backend(expr: TableExpr) -> type[TableImpl]:
    if isinstance(expr, Verb):
        return get_backend(expr.table)
    elif isinstance(expr, Join):
        return get_backend(expr.table)
    else:
        assert isinstance(expr, Table)
        return expr._impl.__class__
