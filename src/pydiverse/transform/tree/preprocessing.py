from __future__ import annotations

import itertools

from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree import col_expr, verbs
from pydiverse.transform.tree.col_expr import Col
from pydiverse.transform.tree.table_expr import TableExpr


# returns the list of cols the table is currently grouped by
def update_partition_by_kwarg(expr: TableExpr):
    if isinstance(expr, verbs.Verb) and not isinstance(expr, verbs.Summarise):
        group_by = update_partition_by_kwarg(expr.table)
        for c in expr.iter_col_roots():
            col_expr.update_partition_by_kwarg(c, group_by)

        if isinstance(expr, verbs.Join):
            update_partition_by_kwarg(expr.right)


# inserts renames before Mutate, Summarise or Join to prevent duplicate column names.
def rename_overwritten_cols(expr: TableExpr):
    if isinstance(expr, verbs.Verb):
        rename_overwritten_cols(expr.table)

        if isinstance(expr, (verbs.Mutate, verbs.Summarise)):
            overwritten = set(name for name in expr.names if name in expr.table._schema)

            if overwritten:
                expr.table = verbs.Rename(
                    expr.table,
                    {name: f"{name}_{str(hash(expr))}" for name in overwritten},
                )

                expr.table = verbs.Drop(
                    expr.table,
                    [Col(name, expr.table) for name in expr.table.name_map.values()],
                )

        if isinstance(expr, verbs.Join):
            rename_overwritten_cols(expr.right)

    else:
        assert isinstance(expr, Table)


# returns Col -> ColName mapping and the list of available columns
def propagate_names(
    expr: TableExpr, needed_cols: set[tuple[TableExpr, str]]
) -> dict[tuple[TableExpr, str], str]:
    if isinstance(expr, verbs.Verb):
        for node in expr.iter_col_nodes():
            if isinstance(node, Col):
                needed_cols.add((node.table, node.name))

        current_name = propagate_names(expr.table, needed_cols)

        if isinstance(expr, verbs.Join):
            current_name |= {
                key: name + expr.suffix
                for key, name in propagate_names(expr.right, needed_cols).items()
            }

        for node in itertools.chain(expr.iter_col_nodes(), expr._group_by):
            if isinstance(node, Col):
                node.name = current_name[(node.table, node.name)]
                node.table = expr.table

        if isinstance(expr, verbs.Rename):
            current_name = {
                key: (expr.name_map[name] if name in expr.name_map else name)
                for key, name in current_name.items()
            }

    elif isinstance(expr, Table):
        current_name = dict()

    else:
        raise AssertionError

    for table, name in needed_cols:
        if expr is table:
            current_name[(expr, name)] = name

    return current_name
