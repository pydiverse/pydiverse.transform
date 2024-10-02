from __future__ import annotations

import copy
import uuid
from collections.abc import Iterable
from typing import Any

from pydiverse.transform.backend.table_impl import TableImpl
from pydiverse.transform.backend.targets import Polars, Target
from pydiverse.transform.errors import FunctionTypeError
from pydiverse.transform.ops.core import Ftype
from pydiverse.transform.pipe.pipeable import builtin_verb
from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree import dtypes, verbs
from pydiverse.transform.tree.ast import AstNode
from pydiverse.transform.tree.col_expr import (
    Col,
    ColExpr,
    ColFn,
    ColName,
    Order,
    wrap_literal,
)
from pydiverse.transform.tree.verbs import (
    Arrange,
    Filter,
    GroupBy,
    Join,
    JoinHow,
    JoinValidate,
    Mutate,
    Rename,
    Select,
    SliceHead,
    Summarise,
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
def alias(table: Table, new_name: str | None = None):
    if new_name is None:
        new_name = table._ast.name
    new = copy.copy(table)
    new._ast, nd_map, uuid_map = table._ast._clone()
    new._ast.name = new_name
    new._cache = copy.copy(table._cache)

    new._cache.partition_by = [
        Col(col.name, nd_map[col._ast], uuid_map[col._uuid], col._dtype, col._ftype)
        for col in table._cache.partition_by
    ]

    new._cache.update(
        new_select=[
            Col(col.name, nd_map[col._ast], uuid_map[col._uuid], col._dtype, col._ftype)
            for col in table._cache.select
        ],
        new_cols={
            name: Col(
                name, nd_map[col._ast], uuid_map[col._uuid], col._dtype, col._ftype
            )
            for name, col in table._cache.cols.items()
        },
    )

    return new


@builtin_verb()
def collect(table: Table, target: Target | None = None) -> Table:
    df = table >> export(Polars(lazy=False))
    if target is None:
        target = Polars()
    return Table(df, target)


@builtin_verb()
def export(table: Table, target: Target):
    check_table_references(table._ast)
    table = table >> alias()
    SourceBackend: type[TableImpl] = get_backend(table._ast)
    return SourceBackend.export(table._ast, target, table._cache.select)


@builtin_verb()
def build_query(table: Table) -> str:
    check_table_references(table._ast)
    table = table >> alias()
    SourceBackend: type[TableImpl] = get_backend(table._ast)
    return SourceBackend.build_query(table._ast, table._cache.select)


@builtin_verb()
def show_query(table: Table):
    if query := table >> build_query():
        print(query)
    else:
        print(f"no query to show for {table._ast.name}")

    return table


@builtin_verb()
def select(table: Table, *args: Col | ColName):
    new = copy.copy(table)
    new._ast = Select(table._ast, preprocess_arg(args, table))
    new._cache = copy.copy(table._cache)
    # TODO: prevent selection of overwritten columns
    new._cache.update(new_select=new._ast.select)
    return new


@builtin_verb()
def drop(table: Table, *args: Col | ColName):
    dropped_uuids = {col._uuid for col in preprocess_arg(args, table)}
    return select(
        table,
        *(col for col in table._cache.select if col._uuid not in dropped_uuids),
    )


@builtin_verb()
def rename(table: Table, name_map: dict[str, str]):
    if not isinstance(name_map, dict):
        raise TypeError("`name_map` argument to `rename` must be a dict")
    if len(name_map) == 0:
        return table

    new = copy.copy(table)
    new._ast = Rename(table._ast, name_map)
    new._cache = copy.copy(table._cache)
    new_cols = copy.copy(table._cache.cols)

    for name, _ in name_map.items():
        if name not in new_cols:
            raise ValueError(
                f"no column with name `{name}` in table `{table._ast.name}`"
            )
        del new_cols[name]

    for name, replacement in name_map.items():
        if replacement in new_cols:
            raise ValueError(f"duplicate column name `{replacement}`")
        new_cols[replacement] = table._cache.cols[name]

    new._cache.update(new_cols=new_cols)

    return new


@builtin_verb()
def mutate(table: Table, **kwargs: ColExpr):
    if len(kwargs) == 0:
        return table

    new = copy.copy(table)
    new._ast = Mutate(
        table._ast,
        list(kwargs.keys()),
        preprocess_arg(kwargs.values(), table),
        [uuid.uuid1() for _ in kwargs.keys()],
    )

    new._cache = copy.copy(table._cache)
    new_cols = copy.copy(table._cache.cols)

    for name, val, uid in zip(new._ast.names, new._ast.values, new._ast.uuids):
        new_cols[name] = Col(
            name, new._ast, uid, val.dtype(), val.ftype(agg_is_window=True)
        )

    overwritten = {
        col_name for col_name in new._ast.names if col_name in table._cache.cols
    }

    new._cache.update(
        new_select=[
            col
            for col in table._cache.select
            if table._cache.uuid_to_name[col._uuid] not in overwritten
        ]
        + [new_cols[name] for name in new._ast.names],
        new_cols=new_cols,
    )

    return new


@builtin_verb()
def filter(table: Table, *predicates: ColExpr):
    if len(predicates) == 0:
        return table

    new = copy.copy(table)
    new._ast = Filter(table._ast, preprocess_arg(predicates, table))

    for cond in new._ast.filters:
        if cond.dtype() != dtypes.Bool:
            raise TypeError(
                "predicates given to `filter` must be of boolean type.\n"
                f"hint: {cond} is of type {cond.dtype()} instead."
            )

    return new


@builtin_verb()
def arrange(table: Table, *order_by: ColExpr):
    if len(order_by) == 0:
        return table

    new = copy.copy(table)
    new._ast = Arrange(
        table._ast,
        preprocess_arg((Order.from_col_expr(ord) for ord in order_by), table),
    )

    return new


@builtin_verb()
def group_by(table: Table, *cols: Col | ColName, add=False):
    if len(cols) == 0:
        return table

    new = copy.copy(table)
    new._ast = GroupBy(table._ast, preprocess_arg(cols, table), add)
    new._cache = copy.copy(table._cache)
    if add:
        new._cache.partition_by = table._cache.partition_by + new._ast.group_by
    else:
        new._cache.partition_by = new._ast.group_by

    return new


@builtin_verb()
def ungroup(table: Table):
    new = copy.copy(table)
    new._ast = Ungroup(table._ast)
    new._cache = copy.copy(table._cache)
    new._cache.partition_by = []
    return new


@builtin_verb()
def summarise(table: Table, **kwargs: ColExpr):
    new = copy.copy(table)
    new._ast = Summarise(
        table._ast,
        list(kwargs.keys()),
        preprocess_arg(kwargs.values(), table, update_partition_by=False),
        [uuid.uuid1() for _ in kwargs.keys()],
    )

    partition_by_uuids = {col._uuid for col in table._cache.partition_by}

    def check_summarise_col_expr(expr: ColExpr, agg_fn_above: bool):
        if (
            isinstance(expr, Col)
            and expr._uuid not in partition_by_uuids
            and not agg_fn_above
        ):
            raise FunctionTypeError(
                f"column `{expr}` is neither aggregated nor part of the grouping "
                "columns."
            )

        elif isinstance(expr, ColFn):
            if expr.ftype(agg_is_window=False) == Ftype.WINDOW:
                raise FunctionTypeError(
                    f"forbidden window function `{expr.name}` in `summarise`"
                )
            elif expr.ftype(agg_is_window=False) == Ftype.AGGREGATE:
                agg_fn_above = True

        for child in expr.iter_children():
            check_summarise_col_expr(child, agg_fn_above)

    for root in new._ast.values:
        check_summarise_col_expr(root, False)

    # TODO: handle duplicate column names
    new._cache = copy.copy(table._cache)

    new_cols = table._cache.cols | {
        name: Col(name, new._ast, uid, val.dtype(), val.ftype(agg_is_window=False))
        for name, val, uid in zip(new._ast.names, new._ast.values, new._ast.uuids)
    }

    new._cache.update(
        new_select=table._cache.partition_by
        + [new_cols[name] for name in new._ast.names],
        new_cols=new_cols,
    )
    new._cache.partition_by = []

    return new


@builtin_verb()
def slice_head(table: Table, n: int, *, offset: int = 0):
    if table._cache.partition_by:
        raise ValueError("cannot apply `slice_head` to a grouped table")

    new = copy.copy(table)
    new._ast = SliceHead(table._ast, n, offset)
    return new


@builtin_verb()
def join(
    left: Table,
    right: Table,
    on: ColExpr,
    how: JoinHow,
    *,
    validate: JoinValidate = "m:m",
    suffix: str | None = None,  # appended to cols of the right table
):
    if left._cache.partition_by:
        raise ValueError(f"cannot join grouped table `{left._ast.name}`")
    elif right._cache.partition_by:
        raise ValueError(f"cannot join grouped table `{right._ast.name}`")

    user_suffix = suffix
    if suffix is None and right._ast.name:
        suffix = f"_{right._ast.name}"
    if suffix is None:
        suffix = "_right"

    left_names = set(left._cache.cols.keys())
    if user_suffix is not None:
        for name in right._cache.cols.keys():
            if name + suffix in left_names:
                raise ValueError(
                    f"column name `{name + suffix}` appears both in the left and right "
                    f"table using the user-provided suffix `{suffix}`\n"
                    "hint: Specify a different suffix to prevent name collisions or "
                    "none at all for automatic name collision resolution."
                )
    else:
        cnt = 0
        for name in right._cache.cols.keys():
            suffixed = name + suffix + (f"_{cnt}" if cnt > 0 else "")
            while suffixed in left_names:
                cnt += 1
                suffixed = name + suffix + f"_{cnt}"

        if cnt > 0:
            suffix += f"_{cnt}"

    new = copy.copy(left)
    new._ast = Join(
        left._ast, right._ast, preprocess_arg(on, left), how, validate, suffix
    )

    new._cache = copy.copy(left._cache)
    new._cache.update(
        new_cols=left._cache.cols
        | {name + suffix: col for name, col in right._cache.cols.items()},
        new_select=left._cache.select + right._cache.select,
    )

    return new


@builtin_verb()
def inner_join(
    left: Table,
    right: Table,
    on: ColExpr,
    *,
    validate: JoinValidate = "m:m",
    suffix: str | None = None,
):
    return left >> join(right, on, "inner", validate=validate, suffix=suffix)


@builtin_verb()
def left_join(
    left: Table,
    right: Table,
    on: ColExpr,
    *,
    validate: JoinValidate = "m:m",
    suffix: str | None = None,
):
    return left >> join(right, on, "left", validate=validate, suffix=suffix)


@builtin_verb()
def outer_join(
    left: Table,
    right: Table,
    on: ColExpr,
    *,
    validate: JoinValidate = "m:m",
    suffix: str | None = None,
):
    return left >> join(right, on, "outer", validate=validate, suffix=suffix)


def preprocess_arg(arg: Any, table: Table, *, update_partition_by: bool = True) -> Any:
    if isinstance(arg, dict):
        return {
            key: preprocess_arg(val, table, update_partition_by=update_partition_by)
            for key, val in arg.items()
        }
    if isinstance(arg, Iterable) and not isinstance(arg, str):
        return [
            preprocess_arg(elem, table, update_partition_by=update_partition_by)
            for elem in arg
        ]
    if isinstance(arg, Order):
        return Order(
            preprocess_arg(
                arg.order_by, table, update_partition_by=update_partition_by
            ),
            arg.descending,
            arg.nulls_last,
        )
    else:
        arg = wrap_literal(arg)
        assert isinstance(arg, ColExpr)

        arg = arg.map_subtree(
            lambda col: col if not isinstance(col, ColName) else table[col.name]
        )

        if not update_partition_by:
            return arg

        from pydiverse.transform.backend.polars import PolarsImpl

        for desc in arg.iter_subtree():
            if (
                isinstance(desc, ColFn)
                and "partition_by" not in desc.context_kwargs
                and (
                    PolarsImpl.registry.get_op(desc.name).ftype
                    in (Ftype.WINDOW, Ftype.AGGREGATE)
                )
            ):
                desc.context_kwargs["partition_by"] = table._cache.partition_by

        return arg


def get_backend(nd: AstNode) -> type[TableImpl]:
    if isinstance(nd, Verb):
        return get_backend(nd.child)
    assert isinstance(nd, TableImpl) and nd is not TableImpl
    return nd.__class__


# checks whether there are duplicate tables and whether all cols used in expressions
# are from descendants
def check_table_references(nd: AstNode) -> set[AstNode]:
    if isinstance(nd, verbs.Verb):
        subtree = check_table_references(nd.child)

        if isinstance(nd, verbs.Join):
            right_tables = check_table_references(nd.right)
            if intersection := subtree & right_tables:
                raise ValueError(
                    f"table `{list(intersection)[0]}` occurs twice in the table "
                    "tree.\nhint: To join two tables derived from a common table, "
                    "apply `>> alias()` to one of them before the join."
                )

            if len(right_tables) > len(subtree):
                subtree, right_tables = right_tables, subtree
            subtree |= right_tables

        for col in nd.iter_col_nodes():
            if isinstance(col, Col) and col._ast not in subtree:
                raise ValueError(
                    f"table `{col._ast.name}` referenced via column `{col}` cannot be "
                    "used at this point. It The current table is not derived "
                    "from it."
                )

        subtree.add(nd)
        return subtree

    else:
        return {nd}
