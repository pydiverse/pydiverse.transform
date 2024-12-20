from __future__ import annotations

import copy
import uuid
from typing import Any, Literal, overload

import pandas as pd
import polars as pl

from pydiverse.transform._internal import errors
from pydiverse.transform._internal.backend.table_impl import TableImpl
from pydiverse.transform._internal.backend.targets import Pandas, Polars, Target
from pydiverse.transform._internal.errors import FunctionTypeError
from pydiverse.transform._internal.ops.op import Ftype
from pydiverse.transform._internal.pipe.c import C
from pydiverse.transform._internal.pipe.pipeable import Pipeable, verb
from pydiverse.transform._internal.pipe.table import Table
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.col_expr import (
    Col,
    ColExpr,
    ColFn,
    ColName,
    Order,
    wrap_literal,
)
from pydiverse.transform._internal.tree.types import Bool
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


@overload
def alias(new_name: str | None = None) -> Pipeable: ...


@verb
def alias(table: Table, new_name: str | None = None) -> Pipeable:
    if new_name is None:
        new_name = table._ast.name
    new = copy.copy(table)
    new._ast, nd_map, uuid_map = table._ast._clone()
    new._ast.name = new_name
    new._ast = Alias(new._ast)
    new._cache = copy.copy(table._cache)

    new._cache.all_cols = {
        uuid_map[uid]: Col(
            col.name, nd_map[col._ast], uuid_map[uid], col._dtype, col._ftype
        )
        for uid, col in table._cache.all_cols.items()
    }
    new._cache.partition_by = [
        new._cache.all_cols[uuid_map[col._uuid]] for col in table._cache.partition_by
    ]

    new._cache._update(
        new_select=[
            new._cache.all_cols[uuid_map[col._uuid]] for col in table._cache.select
        ],
        new_cols={
            name: new._cache.all_cols[uuid_map[col._uuid]]
            for name, col in table._cache.cols.items()
        },
    )

    new._cache.derived_from = set(new._ast.iter_subtree_postorder())

    return new


@overload
def collect(target: Target | None = None) -> Pipeable: ...


@verb
def collect(table: Table, target: Target | None = None) -> Pipeable:
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


@overload
def export(target: Target, *, schema_overrides: dict | None = None) -> Pipeable: ...


@verb
def export(
    data: Table | ColExpr,
    target: Target,
    *,
    schema_overrides: dict[Col, Any] | None = None,
) -> Pipeable:
    """Convert a pydiverse.transform Table to a data frame.

    Parameters
    ----------
    target
        Can currently be either a `Polars` or `Pandas` object. For polars, one can
        specify whether a DataFrame or LazyFrame is returned via the `lazy` kwarg.
        If `lazy=True`, no actual computations are performed, they just get stored in
        the LazyFrame.

    schema_overrides
        A dictionary of columns to backend-specific data types. This controls which data
        types are used when writing to the backend. Because the data types are not
        constrained by pydiverse.transform's type system, this may sometimes be
        preferred over a `cast`.

    Returns
    -------
    A polars or pandas DataFrame / LazyFrame or a series.


    Note
    ----
    This verb can be used with a Table or column expression on the left. When applied to
    a column expression, all tables containing columns in the expression tree are
    gathered and the expression is exported with respect to the most recent table. This
    means that there has to be one column in the expression tree, whose table is a
    common ancestor of all others. If this is not the case, the export fails. Anonymous
    columns (`C`-columns) can be used in the expression and are interpreted with respect
    to this common ancestor table.
    """
    if isinstance(data, ColExpr):
        # Find the common ancestor of all AST nodes of columns appearing in the
        # expression.
        nodes: dict[AstNode, int] = {}
        for col in data.iter_subtree():
            if isinstance(col, Col):
                nodes[col._ast] = 0

        for nd in list(nodes.keys()):
            gen = nd.iter_subtree_preorder()
            try:
                descendant = next(gen)
                while True:
                    if descendant != nd and descendant in nodes:
                        nodes[descendant] += 1
                        descendant = gen.send(True)
                    else:
                        descendant = next(gen)
            except StopIteration:
                ...

        indeg0 = (nd for nd, cnt in nodes.items() if cnt == 0)
        root = next(indeg0)
        try:
            next(indeg0)
            # The AST nodes have a common ancestor if and only if there is exactly one
            # node that has not been reached by another one.
            raise ValueError(
                "cannot export a column expression containing columns without a common "
                "ancestor"
            )
        except StopIteration:
            ...

        uid = uuid.uuid1()
        table = from_ast(root)
        df = (
            table
            >> _mutate([str(uid)], [data], [uid])
            >> select(C[str(uid)])
            >> export(target)
        )

        if isinstance(target, Polars):
            if isinstance(df, pl.LazyFrame):
                df = df.collect()
            return df.get_column(str(uid))
        elif isinstance(target, Pandas):
            assert isinstance(df, pd.DataFrame)
            return pd.Series(df[str(uid)])
        raise AssertionError

    # TODO: allow stuff like pdt.Int(): pl.Uint32() in schema_overrides and resolve that
    # to columns
    SourceBackend: type[TableImpl] = get_backend(data._ast)
    data = data >> alias()
    if schema_overrides is None:
        schema_overrides = {}
    return SourceBackend.export(data._ast, target, data._cache.select, schema_overrides)


@overload
def build_query() -> Pipeable: ...


@verb
def build_query(table: Table) -> Pipeable:
    table = table >> alias()
    SourceBackend: type[TableImpl] = get_backend(table._ast)
    return SourceBackend.build_query(table._ast, table._cache.select)


@overload
def show_query() -> Pipeable: ...


@verb
def show_query(table: Table) -> Pipeable:
    if query := table >> build_query():
        print(query)
    else:
        print(f"no query to show for {table._ast.name}")

    return table


@overload
def select(*cols: Col | ColName) -> Pipeable: ...


@verb
def select(table: Table, *cols: Col | ColName) -> Pipeable:
    errors.check_vararg_type(Col | ColName, "select", *cols)

    new = copy.copy(table)
    new._ast = Select(table._ast, [preprocess_arg(col, table) for col in cols])
    new._cache = copy.copy(table._cache)
    new._cache.update(new._ast)
    # TODO: prevent selection of overwritten columns

    return new


@overload
def drop(*cols: Col | ColName) -> Pipeable: ...


@verb
def drop(table: Table, *cols: Col | ColName) -> Pipeable:
    errors.check_vararg_type(Col | ColName, "drop", *cols)

    dropped_uuids = {preprocess_arg(col, table)._uuid for col in cols}
    return select(
        table,
        *(col for col in table._cache.select if col._uuid not in dropped_uuids),
    )


@overload
def rename(name_map: dict[str, str]) -> Pipeable: ...


@verb
def rename(table: Table, name_map: dict[str, str]) -> Pipeable:
    errors.check_arg_type(dict, "rename", "name_map", name_map)
    if len(name_map) == 0:
        return table

    new = copy.copy(table)
    new._ast = Rename(table._ast, name_map)
    new._cache = copy.copy(table._cache)

    if d := set(name_map) - set(table._cache.cols):
        raise ValueError(
            f"no column with name `{next(iter(d))}` in table `{table._ast.name}`"
        )

    if d := (set(table._cache.cols) - set(name_map)) & set(name_map.values()):
        raise ValueError(f"duplicate column name `{next(iter(d))}`")

    new._cache.update(new._ast)
    return new


@overload
def mutate(**kwargs: ColExpr) -> Pipeable: ...


@verb
def mutate(table: Table, **kwargs: ColExpr) -> Pipeable:
    if len(kwargs) == 0:
        return table
    return table >> _mutate(*map(list, zip(*kwargs.items(), strict=True)))


@verb
def _mutate(
    table: Table,
    names: list[str],
    values: list[ColExpr],
    uuids: list[uuid.UUID] | None = None,
) -> Pipeable:
    new = copy.copy(table)
    if uuids is None:
        uuids = [uuid.uuid1() for _ in names]
    else:
        assert len(uuids) == len(names)

    new._ast = Mutate(
        table._ast,
        names,
        [preprocess_arg(val, table) for val in values],
        uuids,
    )

    new._cache = copy.copy(table._cache)
    new._cache.update(new._ast)

    return new


@overload
def filter(*predicates: ColExpr[Bool]) -> Pipeable: ...


@verb
def filter(table: Table, *predicates: ColExpr[Bool]) -> Pipeable:
    if len(predicates) == 0:
        return table

    new = copy.copy(table)
    new._ast = Filter(table._ast, [preprocess_arg(pred, table) for pred in predicates])

    for cond in new._ast.predicates:
        if not cond.dtype() <= Bool():
            raise TypeError(
                "predicates given to `filter` must be of boolean type.\n"
                f"hint: {cond} is of type {cond.dtype()} instead."
            )

        for fn in cond.iter_subtree():
            if isinstance(fn, ColFn) and fn.op.ftype in (
                Ftype.WINDOW,
                Ftype.AGGREGATE,
            ):
                raise FunctionTypeError(
                    f"forbidden window function `{fn.op.name}` in `filter`\nhint: If "
                    "you want to filter by an expression containing a window / "
                    "aggregation function, first add the expression as a column via "
                    "`mutate`."
                )

    new._cache = copy.copy(table._cache)
    new._cache.update(new._ast)

    return new


@overload
def arrange(*order_by: ColExpr) -> Pipeable: ...


@verb
def arrange(table: Table, *order_by: ColExpr) -> Pipeable:
    if len(order_by) == 0:
        return table

    new = copy.copy(table)
    new._ast = Arrange(
        table._ast,
        [preprocess_arg(Order.from_col_expr(ord), table) for ord in order_by],
    )

    new._cache = copy.copy(table._cache)
    new._cache.update(new._ast)

    return new


@overload
def group_by(table: Table, *cols: Col | ColName, add=False) -> Pipeable: ...


@verb
def group_by(table: Table, *cols: Col | ColName, add=False) -> Pipeable:
    if len(cols) == 0:
        return table

    errors.check_vararg_type(Col | ColName, "group_by", *cols)
    errors.check_arg_type(bool, "group_by", "add", add)

    new = copy.copy(table)
    new._ast = GroupBy(table._ast, [preprocess_arg(col, table) for col in cols], add)
    new._cache = copy.copy(table._cache)
    if add:
        new._cache.partition_by = table._cache.partition_by + new._ast.group_by
    else:
        new._cache.partition_by = new._ast.group_by

    new._cache.derived_from = table._cache.derived_from | {new._ast}

    return new


@overload
def ungroup() -> Pipeable: ...


@verb
def ungroup(table: Table) -> Pipeable:
    new = copy.copy(table)
    new._ast = Ungroup(table._ast)
    new._cache = copy.copy(table._cache)
    new._cache.partition_by = []
    return new


@overload
def summarize(**kwargs: ColExpr) -> Pipeable: ...


@verb
def summarize(table: Table, **kwargs: ColExpr) -> Pipeable:
    return table >> _summarize(*map(list, zip(*kwargs.items(), strict=True)))


@verb
def _summarize(
    table: Table,
    names: list[str],
    values: list[ColExpr],
    uuids: list[uuid.UUID] | None = None,
) -> Pipeable:
    new = copy.copy(table)

    if uuids is None:
        uuids = [uuid.uuid1() for _ in names]
    else:
        assert len(uuids) == len(names)

    new._ast = Summarize(
        table._ast,
        names,
        [preprocess_arg(val, table, update_partition_by=False) for val in values],
        uuids,
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
                    f"forbidden window function `{expr.op.name}` in `summarize`"
                )
            elif expr.ftype(agg_is_window=False) == Ftype.AGGREGATE:
                agg_fn_above = True

        for child in expr.iter_children():
            check_summarize_col_expr(child, agg_fn_above)

    for root in new._ast.values:
        check_summarize_col_expr(root, False)

    new._cache = copy.copy(table._cache)
    new._cache.update(new._ast)

    return new


@overload
def slice_head(n: int, *, offset: int = 0) -> Pipeable: ...


@verb
def slice_head(table: Table, n: int, *, offset: int = 0) -> Pipeable:
    errors.check_arg_type(int, "slice_head", "n", n)
    errors.check_arg_type(int, "slice_head", "offset", offset)

    if table._cache.partition_by:
        raise ValueError("cannot apply `slice_head` to a grouped table")

    new = copy.copy(table)
    new._ast = SliceHead(table._ast, n, offset)
    new._cache = copy.copy(table._cache)
    new._cache.update(new._ast)

    return new


@overload
def join(
    right: Table,
    on: ColExpr[Bool],
    how: Literal["inner", "left", "full"],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable: ...


@verb
def join(
    left: Table,
    right: Table,
    on: ColExpr[Bool],
    how: Literal["inner", "left", "full"],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable:
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

    new = copy.copy(left)
    new._ast = Join(left._ast, right._ast, on, how, validate, suffix)
    # The join condition must be preprocessed with respect to both tables
    new._ast.on = resolve_C_columns(
        resolve_C_columns(new._ast.on, left, strict=False), right, suffix=suffix
    )

    new._cache = copy.copy(left._cache)
    new._cache.update(new._ast, rcache=right._cache)
    new._ast.on = preprocess_arg(new._ast.on, new)

    return new


# We define the join variations explicitly instead of via functools.partial since vscode
# gives functools.partial objects a different color than normal python functions which
# looks very confusing.
@overload
def inner_join(
    right: Table,
    on: ColExpr[Bool],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable: ...


@verb
def inner_join(
    left: Table,
    right: Table,
    on: ColExpr[Bool],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable:
    return left >> join(right, on, "inner", validate=validate, suffix=suffix)


@overload
def left_join(
    right: Table,
    on: ColExpr[Bool],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable: ...


@verb
def left_join(
    left: Table,
    right: Table,
    on: ColExpr[Bool],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable:
    return left >> join(right, on, "left", validate=validate, suffix=suffix)


@overload
def full_join(
    right: Table,
    on: ColExpr[Bool],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable: ...


@verb
def full_join(
    left: Table,
    right: Table,
    on: ColExpr[Bool],
    *,
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:m",
    suffix: str | None = None,
) -> Pipeable:
    return left >> join(right, on, "full", validate=validate, suffix=suffix)


@overload
def show() -> Pipeable: ...


@verb
def show(table: Table) -> Pipeable:
    print(table)
    return table


def resolve_C_columns(expr: ColExpr, table: Table, *, strict=True, suffix=""):
    def resolve_C(col: ColName):
        if strict or col in table:
            return table[col.name[: len(col.name) - len(suffix)]]
        return col

    return expr.map_subtree(
        lambda col: resolve_C(col)
        if isinstance(col, ColName)
        else (
            table._cache.all_cols[col._uuid]
            if isinstance(col, Col) and col._uuid in table._cache.all_cols
            else col
        )
    )


def preprocess_arg(arg: Any, table: Table, *, update_partition_by: bool = True) -> Any:
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
                and (expr.op.ftype in (Ftype.WINDOW, Ftype.AGGREGATE))
            ):
                expr.context_kwargs["partition_by"] = table._cache.partition_by

        return resolve_C_columns(arg, table)


def get_backend(nd: AstNode) -> type[TableImpl]:
    if isinstance(nd, Verb):
        return get_backend(nd.child)
    assert isinstance(nd, TableImpl) and nd is not TableImpl
    return nd.__class__


def from_ast(root: AstNode) -> Table:
    if isinstance(root, TableImpl):
        return Table(root)

    assert isinstance(root, Verb)
    child = from_ast(root.child)
    if isinstance(root, Alias):
        return child >> alias(root.name)

    child._cache.update(
        root, from_ast(root.right)._cache if isinstance(root, Join) else None
    )
    child._ast = root
    return child
