from __future__ import annotations

from pydiverse.transform.tree import verbs
from pydiverse.transform.tree.table_expr import TableExpr


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
