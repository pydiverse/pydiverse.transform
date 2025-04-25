from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any
from uuid import UUID

import polars as pl

from pydiverse.common import (
    Dtype,
    Int,
    String,
)
from pydiverse.transform._internal.backend.table_impl import TableImpl, split_join_cond
from pydiverse.transform._internal.backend.targets import Pandas, Polars, Target
from pydiverse.transform._internal.ops import ops
from pydiverse.transform._internal.ops.op import Ftype
from pydiverse.transform._internal.tree import types, verbs
from pydiverse.transform._internal.tree.ast import AstNode
from pydiverse.transform._internal.tree.col_expr import (
    CaseExpr,
    Cast,
    Col,
    ColExpr,
    ColFn,
    LiteralCol,
    Order,
)


class PolarsImpl(TableImpl):
    backend_name = "polars"

    def __init__(self, name: str, df: pl.DataFrame | pl.LazyFrame):
        self.df = df if isinstance(df, pl.LazyFrame) else df.lazy()
        self.df = self.df.cast(
            {
                pl.Datetime("ns"): pl.Datetime("us"),
                pl.Datetime("ms"): pl.Datetime("us"),
            }
        )
        super().__init__(
            name,
            {
                name: Dtype.from_polars(pl_type)
                for name, pl_type in df.collect_schema().items()
            },
        )

    @staticmethod
    def build_query(nd: AstNode) -> None:
        return None

    @staticmethod
    def export(
        nd: AstNode,
        target: Target,
        *,
        schema_overrides: dict[UUID, Any],  # TODO: use this
    ) -> Any:
        lf, name_in_df, select, _ = compile_ast(nd)
        lf = lf.select(*(name_in_df[uid] for uid in select))

        if isinstance(target, Polars):
            if not target.lazy and isinstance(lf, pl.LazyFrame):
                lf = lf.collect()
            lf.name = nd.name
            return lf

        elif isinstance(target, Pandas):
            return lf.collect().to_pandas(use_pyarrow_extension_array=True)

        raise AssertionError

    def _clone(self) -> tuple[PolarsImpl, dict[AstNode, AstNode], dict[UUID, UUID]]:
        cloned = PolarsImpl(self.name, self.df.clone())
        return (
            cloned,
            {self: cloned},
            {
                self.cols[name]._uuid: cloned.cols[name]._uuid
                for name in self.cols.keys()
            },
        )


# merges descending and null_last markers into the ordering expression
def merge_desc_nulls_last(
    order_by: list[pl.Expr], descending: list[bool], nulls_last: list[bool | None]
) -> list[pl.Expr]:
    merged = []
    for ord, desc, nl in zip(order_by, descending, nulls_last, strict=True):
        # try to avoid this workaround whenever possible
        if nl is not None or desc:
            numeric = ord.rank("dense").cast(pl.Int64)
            if desc:
                numeric = -numeric
            if nl is True:
                numeric = numeric.fill_null(pl.len().cast(pl.Int64) + 1)
            elif nl is False:
                numeric = numeric.fill_null(-pl.len().cast(pl.Int64) - 1)
            merged.append(numeric)
        else:
            merged.append(ord)

    return merged


def compile_order(
    order: Order, name_in_df: dict[UUID, str]
) -> tuple[pl.Expr, bool, bool | None]:
    return (
        compile_col_expr(order.order_by, name_in_df),
        order.descending,
        order.nulls_last,
    )


def compile_col_expr(expr: ColExpr, name_in_df: dict[UUID, str]) -> pl.Expr:
    if isinstance(expr, Col):
        return pl.col(name_in_df[expr._uuid])

    elif isinstance(expr, ColFn):
        impl = PolarsImpl.get_impl(expr.op, tuple(arg.dtype() for arg in expr.args))
        args: list[pl.Expr] = [compile_col_expr(arg, name_in_df) for arg in expr.args]

        if (partition_by := expr.context_kwargs.get("partition_by")) is not None:
            partition_by = [compile_col_expr(pb, name_in_df) for pb in partition_by]

        arrange = expr.context_kwargs.get("arrange")
        if arrange:
            order_by, descending, nulls_last = zip(
                *[compile_order(order, name_in_df) for order in arrange], strict=True
            )

        # The following `if` block is absolutely unnecessary and just an optimization.
        # Otherwise, `over` would be used for sorting, but we cannot pass descending /
        # nulls_last there and the required workaround is probably slower than polars`s
        # native `sort_by`.
        if arrange and not partition_by and len(args) > 0:
            # order the args. if the table is grouped by group_by or
            # partition_by=, the groups will be sorted via over(order_by=)
            # anyways so it need not be done here.
            args[0] = args[0].sort_by(
                by=order_by,
                descending=descending,
                nulls_last=[nl if nl is not None else False for nl in nulls_last],
            )

        if expr.op in (ops.rank, ops.dense_rank):
            assert len(expr.args) == 0
            args = [pl.struct(merge_desc_nulls_last(order_by, descending, nulls_last))]
            arrange = None

        value: pl.Expr = impl(*args, _partition_by=partition_by)

        # TODO: currently, count and list_agg are the only aggregation function where we
        # don't want to return null for cols containing only null values. If this
        # happens for more aggregation functions, make this configurable in e.g. the
        # operator spec.     --> do we want it for str.join??
        if expr.op.ftype == Ftype.AGGREGATE and expr.op not in (
            ops.count_star,
            ops.list_agg,
        ):
            # In `sum` / `any` and other aggregation functions, polars puts a
            # default value (e.g. 0, False) for empty columns, but we want to put
            # Null in this case to let the user decide about the default value via
            # `fill_null` if he likes to set one.
            assert all(types.is_const(arg.dtype()) for arg in expr.args[1:])
            value = pl.when(args[0].count() == 0).then(None).otherwise(value)

        if partition_by:
            # when doing sort_by -> over in polars, for whatever reason the
            # `nulls_last` argument is ignored. thus when both a grouping and an
            # arrangement are specified, we manually add the descending and
            # nulls_last markers to the ordering.
            if arrange:
                order_by = merge_desc_nulls_last(order_by, descending, nulls_last)
            else:
                order_by = None
            value = value.over(partition_by, order_by=order_by)

        elif arrange and expr.op.ftype == Ftype.WINDOW:
            # the function was executed on the ordered arguments. here we
            # restore the original order of the table.
            inv_permutation = pl.int_range(0, pl.len(), dtype=pl.Int64()).sort_by(
                by=order_by,
                descending=descending,
                nulls_last=[nl if nl is not None else False for nl in nulls_last],
            )
            value = value.sort_by(inv_permutation)

        return value

    elif isinstance(expr, CaseExpr):
        assert len(expr.cases) >= 1
        compiled = pl  # to initialize the when/then-chain
        for cond, val in expr.cases:
            compiled = compiled.when(compile_col_expr(cond, name_in_df)).then(
                compile_col_expr(val, name_in_df)
            )
        if expr.default_val is not None:
            compiled = compiled.otherwise(
                compile_col_expr(expr.default_val, name_in_df)
            )
        return compiled

    elif isinstance(expr, LiteralCol):
        return pl.lit(
            expr.val,
            # only give the type explicitly if we can still be sure about it
            # in nested lists with ints / floats mixed we do not give guarantees
            dtype=expr.dtype().to_polars() if types.is_subtype(expr.dtype()) else None,
        )

    elif isinstance(expr, Cast):
        if (
            expr.target_type.is_int() or expr.target_type.is_float()
        ) and types.without_const(expr.val.dtype()) == String():
            expr.val = expr.val.str.strip()
        compiled = compile_col_expr(expr.val, name_in_df).cast(
            expr.target_type.to_polars()
        )

        if (
            types.without_const(expr.val.dtype()).is_float()
            and expr.target_type == String()
        ):
            compiled = compiled.replace("NaN", "nan")

        return compiled

    else:
        raise AssertionError


def rename_overwritten_cols(
    new_names: Iterable[str],
    df: pl.LazyFrame,
    name_in_df: dict[UUID, str],
    *,
    names_to_consider: set[UUID] | None = None,
) -> tuple[pl.LazyFrame, dict[UUID, str]]:
    if names_to_consider is None:
        names_to_consider = set(name_in_df.values())
    overwritten = names_to_consider.intersection(new_names)

    if overwritten:
        name_map = {
            name: f"{name}:{str(hex(uuid.uuid1().int))[2:]}" for name in overwritten
        }
        name_in_df = {
            uid: (name_map[name] if name in name_map else name)
            for uid, name in name_in_df.items()
        }
        df = df.rename(name_map)

    return df, name_in_df


def compile_ast(
    nd: AstNode,
) -> tuple[pl.LazyFrame, dict[UUID, str], list[UUID], list[UUID]]:
    if isinstance(nd, verbs.Verb):
        df, name_in_df, select, partition_by = compile_ast(nd.child)

    if isinstance(nd, verbs.Mutate | verbs.Summarize):
        nd_names_set = set(nd.names)
        # we need to do this before name_in_df is changed
        select = [
            uid
            for uid in (select if isinstance(nd, verbs.Mutate) else partition_by)
            if name_in_df[uid] not in nd_names_set
        ] + nd.uuids

        names_to_consider = (
            set(name_in_df[uid] for uid in partition_by)
            if isinstance(nd, verbs.Summarize)
            else set(name_in_df.values())
        )
        df, name_in_df = rename_overwritten_cols(
            set(name for name in nd.names if name in names_to_consider),
            df,
            name_in_df,
            names_to_consider=names_to_consider,
        )

    if isinstance(nd, verbs.Select):
        select = [col._uuid for col in nd.select]

    elif isinstance(nd, verbs.Rename):
        df = df.rename(nd.name_map)
        name_in_df = {
            uid: (nd.name_map[name] if name in nd.name_map else name)
            for uid, name in name_in_df.items()
        }

    elif isinstance(nd, verbs.Mutate):
        df = df.with_columns(
            **{
                name: compile_col_expr(value, name_in_df)
                for name, value in zip(nd.names, nd.values, strict=True)
            }
        )

        name_in_df.update(
            {uid: name for uid, name in zip(nd.uuids, nd.names, strict=True)}
        )

    elif isinstance(nd, verbs.Filter):
        if nd.predicates:
            df = df.filter(compile_col_expr(fil, name_in_df) for fil in nd.predicates)

    elif isinstance(nd, verbs.Arrange):
        order_by, descending, nulls_last = zip(
            *[compile_order(order, name_in_df) for order in nd.order_by], strict=True
        )
        df = df.sort(
            order_by,
            descending=descending,
            nulls_last=[False if nl is None else nl for nl in nulls_last],
            maintain_order=True,
        )

    elif isinstance(nd, verbs.Summarize):
        # We support usage of aggregated columns in expressions in summarize, but polars
        # creates arrays when doing that. Thus we unwrap the arrays when necessary.
        def has_path_to_leaf_without_agg(expr: ColExpr):
            if isinstance(expr, Col):
                return True
            if isinstance(expr, ColFn) and expr.op.ftype == Ftype.AGGREGATE:
                return False
            return any(
                has_path_to_leaf_without_agg(child) for child in expr.iter_children()
            )

        aggregations = {}
        for name, val in zip(nd.names, nd.values, strict=True):
            compiled = compile_col_expr(val, name_in_df)
            if has_path_to_leaf_without_agg(val):
                compiled = compiled.first()
            aggregations[name] = compiled

        group_by = [name_in_df[uid] for uid in partition_by]
        # polars complains about an empty group_by, so we insert a dummy constant
        # (which will get deselected later)
        df = df.group_by(*(group_by or [pl.lit(0).alias(str(uuid.uuid1()))])).agg(
            **aggregations
        )

        # we have to remove the columns here for the join hidden column rename to work
        # correctly (otherwise it would try to rename hidden columns that do not exist)
        name_in_df = {
            name: uuid for name, uuid in name_in_df.items() if name in partition_by
        }
        name_in_df.update(
            {uid: name for name, uid in zip(nd.names, nd.uuids, strict=True)}
        )
        partition_by = []

    elif isinstance(nd, verbs.SliceHead):
        df = df.slice(nd.offset, nd.n)

    elif isinstance(nd, verbs.GroupBy):
        new_group_by = [col._uuid for col in nd.group_by]
        partition_by = partition_by + new_group_by if nd.add else new_group_by

    elif isinstance(nd, verbs.Ungroup):
        partition_by = []

    elif isinstance(nd, verbs.Join):
        assert len(partition_by) == 0

        right_df, right_name_in_df, right_select, _ = compile_ast(nd.right)
        predicates = split_join_cond(nd.on)

        # check for name collisions among hidden columns
        # we maintain sensible names in the dataframe for all visible columns and try
        # to do so for hidden columns, too. If a hidden column has the same name as a
        # visible column, it gets a hash suffix. If two hidden columns collide, the
        # right one gets a hash. (this is all after applying `nd.suffix`; we don't want
        # to bother the user to provide a suffix only to prevent name collisions of
        # hidden columns)

        right_name_in_df = {
            uid: name + nd.suffix for uid, name in right_name_in_df.items()
        }
        right_df = right_df.rename(
            {name: name + nd.suffix for name in right_df.collect_schema().names()}
        )

        # visible columns
        right_df, right_name_in_df = rename_overwritten_cols(
            set(name_in_df[uid] for uid in select), right_df, right_name_in_df
        )
        df, name_in_df = rename_overwritten_cols(
            set(right_name_in_df[uid] for uid in right_select), df, name_in_df
        )

        # hidden columns
        right_df, right_name_in_df = rename_overwritten_cols(
            set(name_in_df.values()),
            right_df,
            right_name_in_df,
        )

        assert not set(right_name_in_df.keys()) & set(name_in_df.keys())
        name_in_df.update(right_name_in_df)

        eq_predicates = [pred for pred in predicates if pred.op == ops.equal]
        left_on = []
        right_on = []
        for pred in eq_predicates:
            left_on.append(pred.args[0])
            right_on.append(pred.args[1])

            must_swap_cols = None
            for e in pred.args[0].iter_subtree():
                if isinstance(e, Col):
                    must_swap_cols = e._uuid in right_name_in_df
                    assert e._uuid in name_in_df
                    break
            # TODO: find a good place to throw an error if one side of an equality
            # predicate is constant. or do not consider such predicates as equality
            # predicates and put them in join_where
            assert must_swap_cols is not None

            if must_swap_cols:
                left_on[-1], right_on[-1] = right_on[-1], left_on[-1]

        # If there are only equality predicates, use normal join. Else use join_where
        if len(eq_predicates) == len(predicates):
            if len(predicates) == 0:
                # cross join, polars does not like an empty join condition
                df = df.join(right_df, how="cross")

            else:
                df = df.join(
                    right_df,
                    left_on=[compile_col_expr(col, name_in_df) for col in left_on],
                    right_on=[compile_col_expr(col, name_in_df) for col in right_on],
                    how=nd.how,
                    validate=nd.validate,
                    coalesce=False,
                )

        else:
            assert nd.how != "full"
            if nd.how == "left":
                df = df.with_columns(
                    __INDEX__=pl.int_range(0, pl.len(), dtype=pl.Int64)
                )

            joined = df.join_where(
                right_df,
                *(compile_col_expr(pred, name_in_df) for pred in predicates),
            ).with_columns(
                # polars deletes the right column in equality predicates...
                pl.col(name_in_df[left_col._uuid]).alias(name_in_df[right_col._uuid])
                for left_col, right_col in zip(left_on, right_on, strict=True)
                if isinstance(left_col, Col) and isinstance(right_col, Col)
            )

            if nd.how == "left":
                joined = df.join(joined, on="__INDEX__", how="left").drop("__INDEX__")

            df = joined

        select += right_select

    elif isinstance(nd, PolarsImpl):
        df = nd.df
        name_in_df = {col._uuid: col.name for col in nd.cols.values()}
        select = list(name_in_df.keys())
        partition_by = []

    return df, name_in_df, select, partition_by


with PolarsImpl.impl_store.impl_manager as impl:

    @impl(ops.mean)
    def _mean(x):
        return x.mean()

    @impl(ops.min)
    def _min(x):
        return x.min()

    @impl(ops.max)
    def _max(x):
        return x.max()

    @impl(ops.sum)
    def _sum(x):
        return x.sum()

    @impl(ops.all)
    def _all(x):
        return x.all()

    @impl(ops.any)
    def _any(x):
        return x.any()

    @impl(ops.is_null)
    def _is_null(x):
        return x.is_null()

    @impl(ops.is_not_null)
    def _is_not_null(x):
        return x.is_not_null()

    @impl(ops.fill_null)
    def _fill_null(x, y):
        return x.fill_null(y)

    @impl(ops.dt_year)
    def _dt_year(x):
        return x.dt.year().cast(pl.Int64)

    @impl(ops.dt_month)
    def _dt_month(x):
        return x.dt.month().cast(pl.Int64)

    @impl(ops.dt_day)
    def _dt_day(x):
        return x.dt.day().cast(pl.Int64)

    @impl(ops.dt_hour)
    def _dt_hour(x):
        return x.dt.hour().cast(pl.Int64)

    @impl(ops.dt_minute)
    def _dt_minute(x):
        return x.dt.minute().cast(pl.Int64)

    @impl(ops.dt_second)
    def _dt_second(x):
        return x.dt.second().cast(pl.Int64)

    @impl(ops.dt_millisecond)
    def _dt_millisecond(x):
        return x.dt.millisecond().cast(pl.Int64)

    @impl(ops.dt_day_of_week)
    def _dt_day_of_week(x):
        return x.dt.weekday().cast(pl.Int64)

    @impl(ops.dt_day_of_year)
    def _dt_day_of_year(x):
        return x.dt.ordinal_day().cast(pl.Int64)

    @impl(ops.dur_days)
    def _dur_days(x):
        return x.dt.total_days()

    @impl(ops.dur_hours)
    def _dur_hours(x):
        return x.dt.total_hours()

    @impl(ops.dur_minutes)
    def _dur_minutes(x):
        return x.dt.total_minutes()

    @impl(ops.dur_seconds)
    def _dur_seconds(x):
        return x.dt.total_seconds()

    @impl(ops.dur_milliseconds)
    def _dur_milliseconds(x):
        return x.dt.total_milliseconds()

    @impl(ops.row_number)
    def _row_number():
        return pl.int_range(start=1, end=pl.len() + 1, dtype=pl.Int64)

    @impl(ops.rank)
    def _rank(x):
        return x.rank("min").cast(pl.Int64)

    @impl(ops.dense_rank)
    def _dense_rank(x):
        return x.rank("dense").cast(pl.Int64)

    @impl(ops.shift)
    def _shift(x, n, fill_value=None):
        return x.shift(n, fill_value=fill_value)

    @impl(ops.is_in)
    def _is_in(x, *values):
        if len(values) == 0:
            return pl.lit(False)
        return pl.any_horizontal(x == val for val in values)

    @impl(ops.str_contains)
    def _str_contains(x, y):
        return x.str.contains(y)

    @impl(ops.str_starts_with)
    def _str_starts_with(x, y):
        return x.str.starts_with(y)

    @impl(ops.str_ends_with)
    def _str_ends_with(x, y):
        return x.str.ends_with(y)

    @impl(ops.str_lower)
    def _str_lower(x):
        return x.str.to_lowercase()

    @impl(ops.str_upper)
    def _str_upper(x):
        return x.str.to_uppercase()

    @impl(ops.str_replace_all)
    def _str_replace_all(x, to_replace, replacement):
        return x.str.replace_all(to_replace, replacement)

    @impl(ops.str_len)
    def _str_len(x):
        return x.str.len_chars().cast(pl.Int64)

    @impl(ops.str_strip)
    def _str_strip(x):
        return x.str.strip_chars()

    @impl(ops.str_slice)
    def _str_slice(x, offset, length):
        return x.str.slice(offset, length)

    @impl(ops.count)
    def _count(x):
        return x.count().cast(pl.Int64)

    @impl(ops.count_star)
    def _len():
        return pl.len().cast(pl.Int64)

    @impl(ops.horizontal_max)
    def _horizontal_max(*x):
        return pl.max_horizontal(*x)

    @impl(ops.horizontal_min)
    def _horizontal_min(*x):
        return pl.min_horizontal(*x)

    @impl(ops.round)
    def _round(x, digits):
        return x.round(pl.select(digits).item())

    @impl(ops.exp)
    def _exp(x):
        return x.exp()

    @impl(ops.log)
    def _log(x):
        return x.log()

    @impl(ops.floor)
    def _floor(x):
        return x.floor()

    @impl(ops.ceil)
    def _ceil(x):
        return x.ceil()

    @impl(ops.str_to_datetime)
    def _str_to_datetime(x):
        return x.str.to_datetime()

    @impl(ops.str_to_date)
    def _str_to_date(x):
        return x.str.to_date()

    @impl(ops.floordiv)
    def _floordiv(lhs, rhs):
        result_sign = (lhs < 0) ^ (rhs < 0)
        return (abs(lhs) // abs(rhs)) * pl.when(result_sign).then(-1).otherwise(1)
        # TODO: test some alternatives if this is too slow

    @impl(ops.mod)
    def _mod(lhs, rhs):
        return lhs % (abs(rhs) * pl.when(lhs >= 0).then(1).otherwise(-1))
        # TODO: see whether the following is faster:
        # pl.when(lhs >= 0).then(lhs % abs(rhs)).otherwise(lhs % -abs(rhs))

    @impl(ops.is_inf)
    def _is_inf(x):
        return x.is_infinite()

    @impl(ops.is_not_inf)
    def _is_not_inf(x):
        return x.is_finite()

    @impl(ops.is_nan)
    def _is_nan(x):
        return x.is_nan()

    @impl(ops.is_not_nan)
    def _is_not_nan(x):
        return x.is_not_nan()

    @impl(ops.coalesce)
    def _coalesce(*x):
        return pl.coalesce(*x)

    @impl(ops.pow, Int(), Int())
    def _pow(x, y):
        return x.cast(pl.Float64()) ** y

    @impl(ops.str_join)
    def _str_join(x, delim):
        return x.str.join(pl.select(delim).item())

    @impl(ops.prefix_sum)
    def _prefix_sum(x):
        return x.cum_sum()

    @impl(ops.list_agg)
    def _list_agg(x):
        return x
