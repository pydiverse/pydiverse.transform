from __future__ import annotations

import copy
import functools

from pydiverse.transform.ops.core import Ftype
from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree import col_expr, verbs
from pydiverse.transform.tree.col_expr import Col, ColExpr, ColName
from pydiverse.transform.tree.dtypes import Dtype
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

            def rename_col_expr(node: ColExpr):
                if isinstance(node, ColName) and node.name in expr.table.name_map:
                    new_node = copy.copy(node)
                    new_node.name = expr.table.name_map[node.name]
                    return new_node
                return node

            expr.map_col_nodes(rename_col_expr)

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
        for node in expr.iter_col_nodes():
            if isinstance(node, Col):
                needed_cols.add((node.table, node.name))

        col_to_name = propagate_names(expr.table, needed_cols)
        expr.map_col_roots(
            functools.partial(col_expr.propagate_names, col_to_name=col_to_name)
        )

        if isinstance(expr, verbs.Rename):
            col_to_name = {
                key: (expr.name_map[name] if name in expr.name_map else name)
                for key, name in col_to_name.items()
            }

    elif isinstance(expr, verbs.Join):
        for node in expr.on.iter_nodes():
            if isinstance(node, Col):
                needed_cols.add((node.table, node.name))

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


def propagate_types(
    expr: TableExpr,
) -> tuple[dict[str, Dtype], dict[str, Ftype]]:
    if isinstance(expr, (verbs.UnaryVerb)):
        dtype_map, ftype_map = propagate_types(expr.table)
        expr.map_col_roots(
            functools.partial(
                col_expr.propagate_types,
                dtype_map=dtype_map,
                ftype_map=ftype_map,
                agg_is_window=not isinstance(expr, verbs.Summarise),
            )
        )

        if isinstance(expr, verbs.Rename):
            dtype_map = {
                (expr.name_map[name] if name in expr.name_map else name): dtype
                for name, dtype in dtype_map.items()
            }
            ftype_map = {
                (expr.name_map[name] if name in expr.name_map else name): ftype
                for name, ftype in ftype_map.items()
            }

        elif isinstance(expr, (verbs.Mutate, verbs.Summarise)):
            dtype_map.update(
                {name: value.dtype for name, value in zip(expr.names, expr.values)}
            )
            ftype_map.update(
                {name: value.ftype for name, value in zip(expr.names, expr.values)}
            )

    elif isinstance(expr, verbs.Join):
        dtype_map, ftype_map = propagate_types(expr.left)
        right_dtypes, right_ftypes = propagate_types(expr.right)
        dtype_map |= {name + expr.suffix: dtype for name, dtype in right_dtypes.items()}
        ftype_map |= {name + expr.suffix: ftype for name, ftype in right_ftypes.items()}
        expr.on = col_expr.propagate_types(expr.on, dtype_map, ftype_map, False)

    elif isinstance(expr, Table):
        dtype_map = expr.schema
        ftype_map = {name: Ftype.EWISE for name in expr.col_names()}

    else:
        raise AssertionError

    return dtype_map, ftype_map


# returns the list of cols the table is currently grouped by
def update_partition_by_kwarg(expr: TableExpr) -> list[ColExpr]:
    if isinstance(expr, verbs.UnaryVerb) and not isinstance(expr, verbs.Summarise):
        group_by = update_partition_by_kwarg(expr.table)
        for c in expr.iter_col_roots():
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
