from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal

from pydiverse.transform.core.dispatchers import builtin_verb
from pydiverse.transform.core.expressions import (
    Col,
    ColName,
    SymbolicExpression,
)
from pydiverse.transform.core.expressions.expressions import Expr
from pydiverse.transform.core.util import (
    ordered_set,
    sign_peeler,
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

JoinHow = Literal["inner", "left", "outer"]

JoinValidate = Literal["1:1", "1:m", "m:1", "m:m"]


class TableExpr:
    def _validate_verb_level():
        pass


@dataclass
class Alias(TableExpr):
    table: TableExpr
    new_name: str | None


@dataclass
class Select(TableExpr):
    table: TableExpr
    selects: list[Col | ColName]


@dataclass
class Rename(TableExpr):
    table: TableExpr
    name_map: dict[str, str]


@dataclass
class Mutate(TableExpr):
    table: TableExpr
    names: list[str]
    values: list[Expr]


@dataclass
class Join(TableExpr):
    left: TableExpr
    right: TableExpr
    on: Expr
    how: JoinHow
    validate: JoinValidate
    suffix: str | None = None


@dataclass
class Filter(TableExpr):
    table: TableExpr
    filters: list[Expr]


@dataclass
class Summarise(TableExpr):
    table: TableExpr
    names: list[str]
    values: list[Expr]


@dataclass
class Arrange(TableExpr):
    table: TableExpr
    order_by: list[Expr]


@dataclass
class SliceHead(TableExpr):
    table: TableExpr
    n: int
    offset: int


@dataclass
class GroupBy(TableExpr):
    table: TableExpr
    group_by: list[Col | ColName]


@dataclass
class Ungroup(TableExpr):
    table: TableExpr


@builtin_verb()
def alias(table: TableExpr, new_name: str | None = None):
    return Alias(table, new_name)


@builtin_verb()
def collect(table: TableExpr):
    return table.collect()


@builtin_verb()
def export(table: TableExpr):
    table._validate_verb_level()


@builtin_verb()
def build_query(table: TableExpr):
    return table.build_query()


@builtin_verb()
def show_query(table: TableExpr):
    if query := table.build_query():
        print(query)
    else:
        print(f"No query to show for {type(table).__name__}")

    return table


@builtin_verb()
def select(table: TableExpr, *args: Col | ColName):
    return Select(table, list(args))
    if len(args) == 1 and args[0] is Ellipsis:
        # >> select(...)  ->  Select all columns
        args = [
            table.cols[uuid].as_column(name, table)
            for name, uuid in table.named_cols.fwd.items()
        ]

    cols = []
    positive_selection = None
    for col in args:
        col, is_pos = sign_peeler(col)
        if positive_selection is None:
            positive_selection = is_pos
        else:
            if is_pos is not positive_selection:
                raise ValueError(
                    "All columns in input must have the same sign."
                    " Can't mix selection with deselection."
                )

        if not isinstance(col, (Col, ColName)):
            raise TypeError(
                "Arguments to select verb must be of type `Col`'"
                f" and not {type(col)}."
            )
        cols.append(col)

    selects = []
    for col in cols:
        if isinstance(col, Col):
            selects.append(table.named_cols.bwd[col.uuid])
        elif isinstance(col, ColName):
            selects.append(col.name)

    # Invert selection
    if positive_selection is False:
        exclude = set(selects)
        selects.clear()
        for name in table.selects:
            if name not in exclude:
                selects.append(name)

    new_tbl = table.copy()
    new_tbl.preverb_hook("select", *args)
    new_tbl.selects = ordered_set(selects)
    new_tbl.select(*args)
    return new_tbl


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
def mutate(table: TableExpr, **kwargs: Expr):
    return Mutate(table, list(kwargs.keys()), list(kwargs.values()))


@builtin_verb()
def join(
    left: TableExpr,
    right: TableExpr,
    on: Expr,
    how: Literal["inner", "left", "outer"],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,  # appended to cols of the right table
):
    return Join(left, right, on, how, validate, suffix)


inner_join = functools.partial(join, how="inner")
left_join = functools.partial(join, how="left")
outer_join = functools.partial(join, how="outer")


@builtin_verb()
def filter(table: TableExpr, *args: SymbolicExpression):
    return Filter(table, list(args))


@builtin_verb()
def arrange(table: TableExpr, *args: Col):
    return Arrange(table, list(args))


@builtin_verb()
def group_by(table: TableExpr, *args: Col | ColName, add=False):
    return GroupBy(table, list(args), add)


@builtin_verb()
def ungroup(table: TableExpr):
    return Ungroup(table)


@builtin_verb()
def summarise(table: TableExpr, **kwargs: Expr):
    return Summarise(table, list(kwargs.keys()), list(kwargs.values()))


@builtin_verb()
def slice_head(table: TableExpr, n: int, *, offset: int = 0):
    return SliceHead(table, n, offset)
