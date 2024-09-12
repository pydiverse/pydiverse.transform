from __future__ import annotations

from pydiverse.transform.ops.core import Ftype
from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree import verbs
from pydiverse.transform.tree.col_expr import Col, ColFn
from pydiverse.transform.tree.table_expr import TableExpr


# returns the list of cols the table is currently grouped by
def update_partition_by_kwarg(expr: TableExpr):
    if isinstance(expr, verbs.Verb) and not isinstance(expr, verbs.Summarise):
        update_partition_by_kwarg(expr.table)
        for node in expr.iter_col_nodes():
            if isinstance(node, ColFn):
                from pydiverse.transform.backend.polars import PolarsImpl

                impl = PolarsImpl.registry.get_op(node.name)
                if (
                    impl.ftype in (Ftype.WINDOW, Ftype.AGGREGATE)
                    and "partition_by" not in node.context_kwargs
                ):
                    node.context_kwargs["partition_by"] = expr._group_by

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


def propagate_needed_cols(expr: TableExpr):
    if isinstance(expr, verbs.Verb):
        propagate_needed_cols(expr.table)
        if isinstance(expr, verbs.Join):
            propagate_needed_cols(expr.right)

        for node in expr.iter_col_nodes():
            if isinstance(node, Col):
                node.table._needed_cols.append(node)

    else:
        assert isinstance(expr, Table)
