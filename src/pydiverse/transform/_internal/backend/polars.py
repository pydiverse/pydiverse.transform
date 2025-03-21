from __future__ import annotations

from typing import Any
from uuid import UUID

import polars as pl

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
from pydiverse.transform._internal.tree.types import (
    Bool,
    Date,
    Datetime,
    Decimal,
    Dtype,
    Duration,
    Float,
    Float32,
    Float64,
    Int,
    Int8,
    Int16,
    Int32,
    Int64,
    List,
    NullType,
    String,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
)


class PolarsImpl(TableImpl):
    def __init__(self, name: str, df: pl.DataFrame | pl.LazyFrame):
        self.df = df if isinstance(df, pl.LazyFrame) else df.lazy()
        super().__init__(
            name,
            {name: pdt_type(dtype) for name, dtype in df.collect_schema().items()},
        )

    @staticmethod
    def build_query(nd: AstNode, final_select: list[Col]) -> None:
        return None

    @staticmethod
    def export(
        nd: AstNode,
        target: Target,
        final_select: list[Col],
        schema_overrides: dict,
    ) -> Any:
        lf, _, select, _ = compile_ast(nd)
        lf = lf.select(*select)
        if isinstance(target, Polars):
            if not target.lazy:
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

        value: pl.Expr = impl(*args, _pdt_args=expr.args)

        # TODO: currently, count is the only aggregation function where we don't want
        # to return null for cols containing only null values. If this happens for more
        # aggregation functions, make this configurable in e.g. the operator spec.
        if expr.op.ftype == Ftype.AGGREGATE and expr.op != ops.count_star:
            # In `sum` / `any` and other aggregation functions, polars puts a
            # default value (e.g. 0, False) for empty columns, but we want to put
            # Null in this case to let the user decide about the default value via
            # `fill_null` if he likes to set one.
            assert all(arg.dtype().const for arg in expr.args[1:])
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

        elif arrange:
            if expr.op.ftype == Ftype.AGGREGATE:
                # TODO: don't fail, but give a warning that `arrange` is useless
                # here
                ...

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
        return pl.lit(expr.val, dtype=polars_type(expr.dtype()))

    elif isinstance(expr, Cast):
        if (
            expr.target_type <= Int() or expr.target_type <= Float()
        ) and expr.val.dtype() <= String():
            expr.val = expr.val.str.strip()
        compiled = compile_col_expr(expr.val, name_in_df).cast(
            polars_type(expr.target_type)
        )

        if expr.val.dtype() <= Float() and expr.target_type == String():
            compiled = compiled.replace("NaN", "nan")

        return compiled

    else:
        raise AssertionError


def compile_ast(
    nd: AstNode,
) -> tuple[pl.LazyFrame, dict[UUID, str], list[str], list[UUID]]:
    if isinstance(nd, verbs.Verb):
        df, name_in_df, select, partition_by = compile_ast(nd.child)

    if isinstance(nd, verbs.Mutate | verbs.Summarize):
        overwritten = set(name for name in nd.names if name in set(select))
        if overwritten:
            # We rename overwritten cols to some unique dummy name
            name_map = {name: f"{name}_{str(hex(id(nd)))[2:]}" for name in overwritten}
            name_in_df = {
                uid: (name_map[name] if name in name_map else name)
                for uid, name in name_in_df.items()
            }
            df = df.rename(name_map)

        select = [col_name for col_name in select if col_name not in overwritten]

    if isinstance(nd, verbs.Select):
        select = [name_in_df[col._uuid] for col in nd.select]

    elif isinstance(nd, verbs.Rename):
        df = df.rename(nd.name_map)
        name_in_df = {
            uid: (nd.name_map[name] if name in nd.name_map else name)
            for uid, name in name_in_df.items()
        }
        select = [
            nd.name_map[col_name] if col_name in nd.name_map else col_name
            for col_name in select
        ]

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
        select += nd.names

    elif isinstance(nd, verbs.Filter):
        df = df.filter([compile_col_expr(fil, name_in_df) for fil in nd.predicates])

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

        if partition_by:
            df = df.group_by(*(name_in_df[uid] for uid in partition_by)).agg(
                **aggregations
            )
        else:
            df = df.select(**aggregations)

        name_in_df.update(
            {uid: name for name, uid in zip(nd.names, nd.uuids, strict=True)}
        )
        select = [*(name_in_df[uid] for uid in partition_by), *nd.names]
        partition_by = []

    elif isinstance(nd, verbs.SliceHead):
        df = df.slice(nd.offset, nd.n)

    elif isinstance(nd, verbs.GroupBy):
        new_group_by = [col._uuid for col in nd.group_by]
        partition_by = partition_by + new_group_by if nd.add else new_group_by

    elif isinstance(nd, verbs.Ungroup):
        partition_by = []

    elif isinstance(nd, verbs.Join):
        right_df, right_name_in_df, right_select, _ = compile_ast(nd.right)
        assert not set(right_name_in_df.keys()) & set(name_in_df.keys())
        name_in_df.update(
            {uid: name + nd.suffix for uid, name in right_name_in_df.items()}
        )

        assert len(partition_by) == 0

        predicates = split_join_cond(nd.on)
        right_df = right_df.rename(
            {name: name + nd.suffix for name in right_df.collect_schema().names()}
        )

        # Do equality predicates separately because polars coalesces on them.
        eq_predicates = [pred for pred in predicates if pred.op == ops.equal]
        other_predicates = [pred for pred in predicates if pred.op != ops.equal]

        if eq_predicates:
            left_on = []
            right_on = []
            for pred in eq_predicates:
                left_on.append(pred.args[0])
                right_on.append(pred.args[1])

                left_is_left = None
                for e in pred.args[0].iter_subtree():
                    if isinstance(e, Col):
                        left_is_left = e._uuid not in right_name_in_df
                        assert e._uuid in name_in_df
                        break
                assert left_is_left is not None

                if not left_is_left:
                    left_on[-1], right_on[-1] = right_on[-1], left_on[-1]

            df = df.join(
                right_df,
                left_on=[compile_col_expr(col, name_in_df) for col in left_on],
                right_on=[compile_col_expr(col, name_in_df) for col in right_on],
                how=nd.how,
                validate=nd.validate,
                coalesce=nd.coalesce,
            )
        else:
            if nd.how in ("left", "full"):
                df = df.with_columns(
                    __LEFT_INDEX__=pl.int_range(0, pl.len(), dtype=pl.Int64)
                )
            if nd.how in ("full"):
                right_df = right_df.with_columns(
                    __RIGHT_INDEX__=pl.int_range(0, pl.len(), dtype=pl.Int64)
                )

            joined = df.join_where(
                right_df,
                *(compile_col_expr(pred, name_in_df) for pred in other_predicates),
            )

            if nd.how in ("left", "full"):
                joined = df.join(joined, on="__LEFT_INDEX__", how="left").drop(
                    "__LEFT_INDEX__"
                )
            if nd.how in ("full"):
                joined = joined.join(right_df, on="__RIGHT_INDEX__", how="right").drop(
                    "__RIGHT_INDEX__"
                )

            df = joined

        ignore_right = set()
        if nd.coalesce:
            assert all(
                type(pred.args[0]) is Col and type(pred.args[1]) is Col
                for pred in predicates
            )
            ignore_right = {name_in_df[pred.args[0]._uuid] for pred in predicates}
            name_in_df.update(
                {
                    uid: right_name_in_df[uid]
                    for uid, name in name_in_df.items()
                    if uid in right_name_in_df and name in ignore_right
                }
            )

        select += [
            col_name + nd.suffix
            for col_name in right_select
            if col_name not in ignore_right
        ]

    elif isinstance(nd, PolarsImpl):
        df = nd.df
        name_in_df = {col._uuid: col.name for col in nd.cols.values()}
        select = list(nd.cols.keys())
        partition_by = []

    return df, name_in_df, select, partition_by


def pdt_type(pl_type: pl.DataType) -> Dtype:
    if pl_type == pl.Float64():
        return Float64()
    elif pl_type == pl.Float32():
        return Float32()

    elif pl_type == pl.Int64():
        return Int64()
    elif pl_type == pl.Int32():
        return Int32()
    elif pl_type == pl.Int16():
        return Int16()
    elif pl_type == pl.Int8():
        return Int8()

    elif pl_type == pl.UInt64():
        return Uint64()
    elif pl_type == pl.UInt32():
        return Uint32()
    elif pl_type == pl.UInt16():
        return Uint16()
    elif pl_type == pl.UInt8():
        return Uint8()

    elif pl_type.is_decimal():
        return Decimal()
    elif isinstance(pl_type, pl.Boolean):
        return Bool()
    elif isinstance(pl_type, pl.String):
        return String()
    elif isinstance(pl_type, pl.Datetime):
        return Datetime()
    elif isinstance(pl_type, pl.Date):
        return Date()
    elif isinstance(pl_type, pl.Duration):
        return Duration()
    elif isinstance(pl_type, pl.List):
        return List()
    elif isinstance(pl_type, pl.Null):
        return NullType()

    raise TypeError(f"polars type {pl_type} is not supported by pydiverse.transform")


def polars_type(pdt_type: Dtype) -> pl.DataType:
    assert types.is_subtype(pdt_type)

    if pdt_type <= Float64():
        return pl.Float64()
    elif pdt_type <= Float32():
        return pl.Float32()

    elif pdt_type <= Int64():
        return pl.Int64()
    elif pdt_type <= Int32():
        return pl.Int32()
    elif pdt_type <= Int16():
        return pl.Int16()
    elif pdt_type <= Int8():
        return pl.Int8()

    elif pdt_type <= Uint64():
        return pl.UInt64()
    elif pdt_type <= Uint32():
        return pl.UInt32()
    elif pdt_type <= Uint16():
        return pl.UInt16()
    elif pdt_type <= Uint8():
        return pl.UInt8()

    elif pdt_type <= Bool():
        return pl.Boolean()
    elif pdt_type <= String():
        return pl.String()
    elif pdt_type <= Datetime():
        return pl.Datetime()
    elif pdt_type <= Date():
        return pl.Date()
    elif pdt_type <= Duration():
        return pl.Duration()
    elif pdt_type <= List():
        return pl.List
    elif pdt_type <= NullType():
        return pl.Null()

    raise AssertionError


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
