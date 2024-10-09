from __future__ import annotations

import copy
import uuid
from collections.abc import Iterable
from typing import Any, Literal

from pydiverse.transform._internal import errors
from pydiverse.transform._internal.backend.table_impl import TableImpl
from pydiverse.transform._internal.backend.targets import Polars, Target
from pydiverse.transform._internal.errors import FunctionTypeError
from pydiverse.transform._internal.ops.core import Ftype
from pydiverse.transform._internal.pipe.pipeable import verb
from pydiverse.transform._internal.pipe.table import Table
from pydiverse.transform._internal.tree import dtypes
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.col_expr import (
    Col,
    ColExpr,
    ColFn,
    ColName,
    Order,
    wrap_literal,
)
from pydiverse.transform._internal.tree.verbs import (
    Alias,
    Arrange,
    Filter,
    GroupBy,
    Join,
    Mutate,
    Rename,
    Select,
    SliceHead,
    Summarize,
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
    "full_join",
    "filter",
    "arrange",
    "group_by",
    "ungroup",
    "summarize",
    "slice_head",
    "export",
]


@verb
def alias(table: Table, new_name: str | None = None):
    if new_name is None:
        new_name = table._ast.name
    new = copy.copy(table)
    new._ast, nd_map, uuid_map = table._ast._clone()
    new._ast.name = new_name
    new._ast = Alias(new._ast)
    new._cache = copy.copy(table._cache)

    # Why do we copy everything here? column UUIDs have to be rerolled => column
    # expressions need to be rebuilt => verb nodes need to be rebuilt

    # TODO: think about more efficient ways. We could e.g. just update the _cache UUIDs
    # and do the UUID reroll on export (that we do anyway, currently, to allow the
    # backends to rewrite the tree). Then each TableImpl would have to carry some hash
    # for self-join detection.
    # We could also do lazy alias, e.g. wait until a join happens and then only copy
    # the common subtree.

    new._cache.all_cols = {
        uuid_map[uid]: Col(
            col.name, nd_map[col._ast], uuid_map[uid], col._dtype, col._ftype
        )
        for uid, col in table._cache.all_cols.items()
    }
    new._cache.partition_by = [
        new._cache.all_cols[uuid_map[col._uuid]] for col in table._cache.partition_by
    ]

    new._cache.update(
        new_select=[
            new._cache.all_cols[uuid_map[col._uuid]] for col in table._cache.select
        ],
        new_cols={
            name: new._cache.all_cols[uuid_map[col._uuid]]
            for name, col in table._cache.cols.items()
        },
    )

    new._cache.derived_from = set(new._ast.iter_subtree())

    return new


@verb
def collect(table: Table, target: Target | None = None) -> Table:
    errors.check_arg_type(Target | None, "collect", "target", target)

    df = table >> select(*table._cache.all_cols.values()) >> export(Polars(lazy=False))
    if target is None:
        target = Polars()

    new = Table(
        TableImpl.from_resource(
            df,
            target,
            name=table._ast.name,
            # preserve UUIDs and by this column references across collect()
            uuids={name: col._uuid for name, col in table._cache.cols.items()},
        )
    )
    new._cache.derived_from = table._cache.derived_from | {new._ast}
    new._cache.select = [preprocess_arg(col, new) for col in table._cache.select]
    new._cache.partition_by = [
        preprocess_arg(col, new) for col in table._cache.partition_by
    ]
    return new


@verb
def export(table: Table, target: Target):
    table = table >> alias()
    SourceBackend: type[TableImpl] = get_backend(table._ast)
    return SourceBackend.export(table._ast, target, table._cache.select)


@verb
def build_query(table: Table) -> str:
    table = table >> alias()
    SourceBackend: type[TableImpl] = get_backend(table._ast)
    return SourceBackend.build_query(table._ast, table._cache.select)


@verb
def show_query(table: Table):
    if query := table >> build_query():
        print(query)
    else:
        print(f"no query to show for {table._ast.name}")

    return table


@verb
def select(table: Table, *cols: Col | ColName):
    errors.check_vararg_type(Col | ColName, "select", *cols)

    new = copy.copy(table)
    new._ast = Select(table._ast, preprocess_arg(cols, table))
    new._cache = copy.copy(table._cache)
    # TODO: prevent selection of overwritten columns
    new._cache.update(new_select=new._ast.select)
    new._cache.derived_from = table._cache.derived_from | {new._ast}

    return new


@verb
def drop(table: Table, *cols: Col | ColName):
    errors.check_vararg_type(Col | ColName, "drop", *cols)

    dropped_uuids = {col._uuid for col in preprocess_arg(cols, table)}
    return select(
        table,
        *(col for col in table._cache.select if col._uuid not in dropped_uuids),
    )


@verb
def rename(table: Table, name_map: dict[str, str]):
    errors.check_arg_type(dict, "rename", "name_map", name_map)
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
    new._cache.derived_from = table._cache.derived_from | {new._ast}

    return new


@verb
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

    for name, val, uid in zip(
        new._ast.names, new._ast.values, new._ast.uuids, strict=True
    ):
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
    new._cache.derived_from = table._cache.derived_from | {new._ast}

    return new


@verb
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

        for fn in cond.iter_subtree():
            if isinstance(fn, ColFn) and fn.op().ftype in (
                Ftype.WINDOW,
                Ftype.AGGREGATE,
            ):
                raise FunctionTypeError(
                    f"forbidden window function `{fn.name}` in `filter`\nhint: If you "
                    "want to filter by an expression containing a window / aggregation "
                    "function, first add the expression as a column via `mutate`."
                )

    new._cache = copy.copy(table._cache)
    new._cache.derived_from = table._cache.derived_from | {new._ast}

    return new


@verb
def arrange(table: Table, *order_by: ColExpr):
    if len(order_by) == 0:
        return table

    new = copy.copy(table)
    new._ast = Arrange(
        table._ast,
        preprocess_arg((Order.from_col_expr(ord) for ord in order_by), table),
    )

    new._cache.derived_from = table._cache.derived_from | {new._ast}

    return new


@verb
def group_by(table: Table, *cols: Col | ColName, add=False):
    if len(cols) == 0:
        return table

    errors.check_vararg_type(Col | ColName, "group_by", *cols)
    errors.check_arg_type(bool, "group_by", "add", add)

    new = copy.copy(table)
    new._ast = GroupBy(table._ast, preprocess_arg(cols, table), add)
    new._cache = copy.copy(table._cache)
    if add:
        new._cache.partition_by = table._cache.partition_by + new._ast.group_by
    else:
        new._cache.partition_by = new._ast.group_by

    new._cache.derived_from = table._cache.derived_from | {new._ast}

    return new


@verb
def ungroup(table: Table):
    new = copy.copy(table)
    new._ast = Ungroup(table._ast)
    new._cache = copy.copy(table._cache)
    new._cache.partition_by = []
    return new


@verb
def summarize(table: Table, **kwargs: ColExpr):
    new = copy.copy(table)
    new._ast = Summarize(
        table._ast,
        list(kwargs.keys()),
        preprocess_arg(kwargs.values(), table, update_partition_by=False),
        [uuid.uuid1() for _ in kwargs.keys()],
    )

    partition_by_uuids = {col._uuid for col in table._cache.partition_by}

    def check_summarize_col_expr(expr: ColExpr, agg_fn_above: bool):
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
                    f"forbidden window function `{expr.name}` in `summarize`"
                )
            elif expr.ftype(agg_is_window=False) == Ftype.AGGREGATE:
                agg_fn_above = True

        for child in expr.iter_children():
            check_summarize_col_expr(child, agg_fn_above)

    for root in new._ast.values:
        check_summarize_col_expr(root, False)

    # TODO: handle duplicate column names
    new._cache = copy.copy(table._cache)

    new_cols = table._cache.cols | {
        name: Col(name, new._ast, uid, val.dtype(), val.ftype(agg_is_window=False))
        for name, val, uid in zip(
            new._ast.names, new._ast.values, new._ast.uuids, strict=True
        )
    }

    new._cache.update(
        new_select=table._cache.partition_by
        + [new_cols[name] for name in new._ast.names],
        new_cols=new_cols,
    )
    new._cache.partition_by = []
    new._cache.derived_from = table._cache.derived_from | {new._ast}

    return new


@verb
def slice_head(table: Table, n: int, *, offset: int = 0):
    errors.check_arg_type(int, "slice_head", "n", n)
    errors.check_arg_type(int, "slice_head", "offset", offset)

    if table._cache.partition_by:
        raise ValueError("cannot apply `slice_head` to a grouped table")

    new = copy.copy(table)
    new._ast = SliceHead(table._ast, n, offset)
    new._cache = copy.copy(table._cache)
    new._cache.derived_from = table._cache.derived_from | {new._ast}

    return new


@verb
def join(
    left: Table,
    right: Table,
    on: ColExpr,
    how: Literal["inner", "left", "full"],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
):
    errors.check_arg_type(Table, "join", "right", right)
    errors.check_arg_type(str | None, "join", "suffix", suffix)
    errors.check_literal_type(["inner", "left", "full"], "join", "how", how)
    errors.check_literal_type(
        ["1:1", "1:m", "m:1", "m:m"], "join", "validate", validate
    )

    if left._cache.partition_by:
        raise ValueError(f"cannot join grouped table `{left._ast.name}`")
    elif right._cache.partition_by:
        raise ValueError(f"cannot join grouped table `{right._ast.name}`")

    if intersection := left._cache.derived_from & right._cache.derived_from:
        raise ValueError(
            f"table `{list(intersection)[0]}` occurs twice in the table "
            "tree.\nhint: To join two tables derived from a common table, "
            "apply `>> alias()` to one of them before the join."
        )

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

    # The arg preprocessing for join is a bit more complicated since we have to give the
    # joined table to `preprocess_args` so that C.<right column> works.
    new = copy.copy(left)
    new._ast = Join(left._ast, right._ast, on, how, validate, suffix)

    new._cache = copy.copy(left._cache)
    new._cache.update(
        new_cols=left._cache.cols
        | {name + suffix: col for name, col in right._cache.cols.items()},
        new_select=left._cache.select + right._cache.select,
    )
    new._cache.derived_from = (
        left._cache.derived_from | right._cache.derived_from | {new._ast}
    )
    new._ast.on = preprocess_arg(new._ast.on, new, update_partition_by=False)

    return new


# We define the join variations explicitly instead of via functools.partial since vscode
# gives functools.partial objects a different color than normal python functions which
# looks very confusing.
@verb
def inner_join(
    left: Table,
    right: Table,
    on: ColExpr,
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
):
    return left >> join(right, on, "inner", validate=validate, suffix=suffix)


@verb
def left_join(
    left: Table,
    right: Table,
    on: ColExpr,
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
):
    return left >> join(right, on, "left", validate=validate, suffix=suffix)


@verb
def full_join(
    left: Table,
    right: Table,
    on: ColExpr,
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
):
    return left >> join(right, on, "full", validate=validate, suffix=suffix)


@verb
def show(table: Table):
    print(table)
    return table


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

        for expr in arg.iter_subtree():
            if isinstance(expr, Col) and expr._ast not in table._cache.derived_from:
                raise ValueError(
                    f"table `{expr._ast.name}` used to reference the column "
                    f"`{repr(expr)}` cannot be used at this point. The current table "
                    "is not derived from it."
                )
            if (
                update_partition_by
                and isinstance(expr, ColFn)
                and "partition_by" not in expr.context_kwargs
                and (expr.op().ftype in (Ftype.WINDOW, Ftype.AGGREGATE))
            ):
                expr.context_kwargs["partition_by"] = table._cache.partition_by

        arg: ColExpr = arg.map_subtree(
            lambda col: table[col.name]
            if isinstance(col, ColName)
            else (table._cache.all_cols[col._uuid] if isinstance(col, Col) else col)
        )

        return arg


def get_backend(nd: AstNode) -> type[TableImpl]:
    if isinstance(nd, Verb):
        return get_backend(nd.child)
    assert isinstance(nd, TableImpl) and nd is not TableImpl
    return nd.__class__
