from __future__ import annotations

import functools

from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree import col_expr, dtypes, verbs
from pydiverse.transform.tree.col_expr import ColExpr, ColName
from pydiverse.transform.tree.table_expr import TableExpr


# inserts renames before Mutate, Summarise or Join to prevent duplicate column names.
def rename_overwritten_cols(expr: TableExpr) -> tuple[set[str], list[str]]:
    if isinstance(expr, verbs.UnaryVerb) and not isinstance(
        expr, (verbs.Mutate, verbs.Summarise, verbs.GroupBy, verbs.Ungroup)
    ):
        return rename_overwritten_cols(expr.table)

    elif isinstance(expr, (verbs.Mutate, verbs.Summarise)):
        available_cols, group_by = rename_overwritten_cols(expr.table)
        if isinstance(expr, verbs.Summarise):
            available_cols = set(group_by)
        overwritten = set(name for name in expr.names if name in available_cols)

        if overwritten:
            expr.table = verbs.Rename(
                expr.table, {name: f"{name}_{str(hash(expr))}" for name in overwritten}
            )
            for val in expr.values:
                col_expr.rename_overwritten_cols(val, expr.table.name_map)
            expr.table = verbs.Drop(
                expr.table, [ColName(name) for name in expr.table.name_map.values()]
            )

        available_cols |= set(
            {
                (name if name not in overwritten else f"{name}_{str(hash(expr))}")
                for name in expr.names
            }
        )

    elif isinstance(expr, verbs.GroupBy):
        available_cols, group_by = rename_overwritten_cols(expr.table)
        group_by = expr.group_by + group_by if expr.add else expr.group_by

    elif isinstance(expr, verbs.Ungroup):
        available_cols, _ = rename_overwritten_cols(expr.table)
        group_by = []

    elif isinstance(expr, verbs.Join):
        left_available, _ = rename_overwritten_cols(expr.left)
        right_avaialable, _ = rename_overwritten_cols(expr.right)
        available_cols = left_available | set(
            {name + expr.suffix for name in right_avaialable}
        )
        group_by = []

    elif isinstance(expr, Table):
        available_cols = set(expr.col_names())
        group_by = []

    else:
        raise AssertionError

    return available_cols, group_by


# returns Col -> ColName mapping and the list of available columns
def propagate_names(
    expr: TableExpr, needed_cols: set[tuple[TableExpr, str]]
) -> dict[tuple[TableExpr, str], str]:
    if isinstance(expr, verbs.UnaryVerb):
        for c in expr.col_exprs():
            needed_cols |= col_expr.get_needed_cols(c)
        col_to_name = propagate_names(expr.table, needed_cols)
        expr.replace_col_exprs(
            functools.partial(col_expr.propagate_names, col_to_name=col_to_name)
        )

        if isinstance(expr, verbs.Rename):
            col_to_name = {
                key: (expr.name_map[name] if name in expr.name_map else name)
                for key, name in col_to_name.items()
            }

    elif isinstance(expr, verbs.Join):
        needed_cols |= col_expr.get_needed_cols(expr.on)
        col_to_name = propagate_names(expr.left, needed_cols)
        col_to_name_right = propagate_names(expr.right, needed_cols)
        col_to_name |= {
            key: name + expr.suffix for key, name in col_to_name_right.items()
        }
        expr.on = col_expr.propagate_names(expr.on, col_to_name)

    elif isinstance(expr, Table):
        col_to_name = dict()

    else:
        raise AssertionError

    for table, name in needed_cols:
        if expr is table:
            col_to_name[(expr, name)] = name

    return col_to_name


def propagate_types(expr: TableExpr) -> dict[str, dtypes.DType]:
    if isinstance(expr, (verbs.UnaryVerb)):
        col_types = propagate_types(expr.table)
        expr.replace_col_exprs(
            functools.partial(col_expr.propagate_types, col_types=col_types)
        )

        if isinstance(expr, verbs.Rename):
            col_types = {
                (expr.name_map[name] if name in expr.name_map else name): dtype
                for name, dtype in propagate_types(expr.table).items()
            }

        elif isinstance(expr, (verbs.Mutate, verbs.Summarise)):
            col_types.update(
                {name: value.dtype for name, value in zip(expr.names, expr.values)}
            )

    elif isinstance(expr, verbs.Join):
        col_types = propagate_types(expr.left) | {
            name + expr.suffix: dtype
            for name, dtype in propagate_types(expr.right).items()
        }
        expr.on = col_expr.propagate_types(expr.on, col_types)

    elif isinstance(expr, Table):
        col_types = expr.schema

    else:
        raise AssertionError

    return col_types


# returns the list of cols the table is currently grouped by
def update_partition_by_kwarg(expr: TableExpr) -> list[ColExpr]:
    if isinstance(expr, verbs.UnaryVerb) and not isinstance(expr, verbs.Summarise):
        group_by = update_partition_by_kwarg(expr.table)
        for c in expr.col_exprs():
            col_expr.update_partition_by_kwarg(c, group_by)

        if isinstance(expr, verbs.GroupBy):
            group_by = expr.group_by

        elif isinstance(expr, verbs.Ungroup):
            group_by = []

    elif isinstance(expr, verbs.Join):
        update_partition_by_kwarg(expr.left)
        update_partition_by_kwarg(expr.right)
        group_by = []

    elif isinstance(expr, (verbs.Summarise, Table)):
        group_by = []

    else:
        raise AssertionError

    return group_by
