from __future__ import annotations

from pydiverse.transform.ops.core import Ftype
from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree import verbs
from pydiverse.transform.tree.col_expr import Col, ColExpr, ColFn, ColName
from pydiverse.transform.tree.table_expr import TableExpr


# returns the list of cols the table is currently grouped by
def update_partition_by_kwarg(expr: TableExpr):
    if isinstance(expr, verbs.Verb):
        update_partition_by_kwarg(expr.table)

        if not isinstance(expr, verbs.Summarise):
            for node in expr.iter_col_nodes():
                if isinstance(node, ColFn):
                    from pydiverse.transform.backend.polars import PolarsImpl

                    impl = PolarsImpl.registry.get_op(node.name)
                    if (
                        impl.ftype in (Ftype.WINDOW, Ftype.AGGREGATE)
                        and "partition_by" not in node.context_kwargs
                    ):
                        node.context_kwargs["partition_by"] = expr.table._partition_by

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

                for node in expr.iter_col_nodes():
                    if isinstance(node, ColName) and node.name in expr.table.name_map:
                        node.name = expr.table.name_map[node.name]

                expr.table = verbs.Drop(
                    expr.table,
                    [ColName(name) for name in expr.table.name_map.values()],
                )

        if isinstance(expr, verbs.Join):
            rename_overwritten_cols(expr.right)

    else:
        assert isinstance(expr, Table)


def propagate_names(expr: TableExpr, needed_cols: set[Col]) -> dict[Col, ColName]:
    if isinstance(expr, verbs.Verb):
        for node in expr.iter_col_nodes():
            if isinstance(node, Col):
                needed_cols.add(node)

        # in summarise, the grouping columns are added to the table, so we need to
        # resolve them
        if isinstance(expr, verbs.Summarise):
            needed_cols.update(expr.table._partition_by)

        col_to_name = propagate_names(expr.table, needed_cols)

        if isinstance(expr, verbs.Join):
            col_to_name_right = propagate_names(expr.right, needed_cols)
            for _, col in col_to_name_right.items():
                col.name += expr.suffix
            col_to_name |= col_to_name_right

        def replace_cols(node: ColExpr) -> ColExpr:
            if isinstance(node, Col):
                if (replacement := col_to_name.get(node)) is None:
                    raise ValueError(
                        f"invalid usage of column `{repr(node)}` in an expression not "
                        f"derived from the table `{node.table.name}`"
                    )
                return replacement
            return node

        expr.map_col_nodes(replace_cols)
        if isinstance(expr, verbs.Summarise):
            expr.table._partition_by = [
                pb.map_nodes(replace_cols) for pb in expr.table._partition_by
            ]

        elif isinstance(expr, verbs.Rename):
            col_to_name.update(
                {
                    col: ColName(
                        expr.name_map[col_name.name], col_name._dtype, col_name._ftype
                    )
                    for col, col_name in col_to_name.items()
                    if col_name.name in expr.name_map
                }
            )

    elif isinstance(expr, Table):
        col_to_name = dict()

    # TODO: use dict[dict] for needed_cols for better efficiency
    for col in needed_cols:
        if col.table is expr:
            col_to_name[col] = ColName(
                col.name,
                col.dtype(),
                col.ftype(agg_is_window=not isinstance(expr, verbs.Summarise)),
            )

    return col_to_name


def check_duplicate_tables(expr: TableExpr) -> set[TableExpr]:
    if isinstance(expr, verbs.Verb):
        tables = check_duplicate_tables(expr.table)

        if isinstance(expr, verbs.Join):
            right_tables = check_duplicate_tables(expr.right)
            if intersection := tables & right_tables:
                raise ValueError(
                    f"table `{list(intersection)[0]}` occurs twice in the table "
                    "tree.\nhint: To join two tables derived from a common table, "
                    "apply `>> alias()` to one of them before the join."
                )

            if len(right_tables) > len(tables):
                tables, right_tables = right_tables, tables
            tables |= right_tables

        return tables

    else:
        return {expr}
