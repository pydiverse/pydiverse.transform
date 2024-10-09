from __future__ import annotations

from typing import Any
from uuid import UUID

import polars as pl

from pydiverse.transform._internal import ops
from pydiverse.transform._internal.backend.table_impl import TableImpl
from pydiverse.transform._internal.backend.targets import Polars, Target
from pydiverse.transform._internal.ops.core import Ftype
from pydiverse.transform._internal.tree import dtypes, verbs
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
    def __init__(self, name: str, df: pl.DataFrame | pl.LazyFrame):
        self.df = df if isinstance(df, pl.LazyFrame) else df.lazy()
        super().__init__(
            name,
            {
                name: polars_type_to_pdt(dtype)
                for name, dtype in df.collect_schema().items()
            },
        )

    @staticmethod
    def build_query(nd: AstNode, final_select: list[Col]) -> None:
        return None

    @staticmethod
    def export(nd: AstNode, target: Target, final_select: list[Col]) -> Any:
        lf, _, select, _ = compile_ast(nd)
        lf = lf.select(select)
        if isinstance(target, Polars):
            if not target.lazy:
                lf = lf.collect()
            lf.name = nd.name
            return lf

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
        op = PolarsImpl.registry.get_op(expr.name)
        args: list[pl.Expr] = [compile_col_expr(arg, name_in_df) for arg in expr.args]
        impl = PolarsImpl.registry.get_impl(
            expr.name,
            tuple(arg.dtype() for arg in expr.args),
        )

        if (partition_by := expr.context_kwargs.get("partition_by")) is not None:
            partition_by = [compile_col_expr(pb, name_in_df) for pb in partition_by]

        arrange = expr.context_kwargs.get("arrange")
        if arrange:
            order_by, descending, nulls_last = zip(
                *[compile_order(order, name_in_df) for order in arrange], strict=True
            )

        # The following `if` block is absolutely unecessary and just an optimization.
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

        if op.name in ("rank", "dense_rank"):
            assert len(expr.args) == 0
            args = [pl.struct(merge_desc_nulls_last(order_by, descending, nulls_last))]
            arrange = None

        value: pl.Expr = impl(*args)

        # TODO: currently, count is the only aggregation function where we don't want
        # to return null for cols containing only null values. If this happens for more
        # aggregation functions, make this configurable in e.g. the operator spec.
        if op.ftype == Ftype.AGGREGATE and op.name != "count":
            # In `sum` / `any` and other aggregation functions, polars puts a
            # default value (e.g. 0, False) for empty columns, but we want to put
            # Null in this case to let the user decide about the default value via
            # `fill_null` if he likes to set one.
            assert all(arg.dtype().const for arg in expr.args[1:])
            value = pl.when(args[0].count() == 0).then(None).otherwise(value)

        if partition_by:
            # when doing sort_by -> over in polars, for whatever reason the
            # `nulls_last` argument is ignored. thus when both a grouping and an
            # arrangment are specified, we manually add the descending and
            # nulls_last markers to the ordering.
            if arrange:
                order_by = merge_desc_nulls_last(order_by, descending, nulls_last)
            else:
                order_by = None
            value = value.over(partition_by, order_by=order_by)

        elif arrange:
            if op.ftype == Ftype.AGGREGATE:
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
        return pl.lit(expr.val, dtype=pdt_type_to_polars(expr.dtype()))

    elif isinstance(expr, Cast):
        compiled = compile_col_expr(expr.val, name_in_df).cast(
            pdt_type_to_polars(expr.target_type)
        )

        if expr.val.dtype() == dtypes.Float64 and expr.target_type == dtypes.String:
            compiled = compiled.replace("NaN", "nan")

        return compiled

    else:
        raise AssertionError


def compile_join_cond(
    expr: ColExpr, name_in_df: dict[UUID, str]
) -> list[tuple[pl.Expr, pl.Expr]]:
    if isinstance(expr, ColFn):
        if expr.name == "__and__":
            return compile_join_cond(expr.args[0], name_in_df) + compile_join_cond(
                expr.args[1], name_in_df
            )
        if expr.name == "__eq__":
            return [
                (
                    compile_col_expr(expr.args[0], name_in_df),
                    compile_col_expr(expr.args[1], name_in_df),
                )
            ]

    raise AssertionError()


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
        df = df.filter([compile_col_expr(fil, name_in_df) for fil in nd.filters])

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
            if isinstance(expr, ColFn) and expr.op().ftype == Ftype.AGGREGATE:
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
        name_in_df.update(
            {uid: name + nd.suffix for uid, name in right_name_in_df.items()}
        )
        left_on, right_on = zip(*compile_join_cond(nd.on, name_in_df), strict=True)

        assert len(partition_by) == 0
        select += [col_name + nd.suffix for col_name in right_select]

        df = df.join(
            right_df.rename({name: name + nd.suffix for name in right_df.columns}),
            left_on=left_on,
            right_on=right_on,
            how=nd.how,
            validate=nd.validate,
            coalesce=False,
        )

    elif isinstance(nd, PolarsImpl):
        df = nd.df
        name_in_df = {col._uuid: col.name for col in nd.cols.values()}
        select = list(nd.cols.keys())
        partition_by = []

    return df, name_in_df, select, partition_by


def polars_type_to_pdt(t: pl.DataType) -> dtypes.Dtype:
    if t.is_float():
        return dtypes.Float64()
    elif t.is_integer():
        return dtypes.Int64()
    elif isinstance(t, pl.Boolean):
        return dtypes.Bool()
    elif isinstance(t, pl.String):
        return dtypes.String()
    elif isinstance(t, pl.Datetime):
        return dtypes.DateTime()
    elif isinstance(t, pl.Date):
        return dtypes.Date()
    elif isinstance(t, pl.Duration):
        return dtypes.Duration()
    elif isinstance(t, pl.Null):
        return dtypes.NoneDtype()

    raise TypeError(f"polars type {t} is not supported by pydiverse.transform")


def pdt_type_to_polars(t: dtypes.Dtype) -> pl.DataType:
    if isinstance(t, dtypes.Float64 | dtypes.Decimal):
        return pl.Float64()
    elif isinstance(t, dtypes.Int64):
        return pl.Int64()
    elif isinstance(t, dtypes.Bool):
        return pl.Boolean()
    elif isinstance(t, dtypes.String):
        return pl.String()
    elif isinstance(t, dtypes.DateTime):
        return pl.Datetime()
    elif isinstance(t, dtypes.Date):
        return pl.Date()
    elif isinstance(t, dtypes.Duration):
        return pl.Duration()
    elif isinstance(t, dtypes.NoneDtype):
        return pl.Null()

    raise AssertionError


with PolarsImpl.op(ops.Mean()) as op:

    @op.auto
    def _mean(x):
        return x.mean()


with PolarsImpl.op(ops.Min()) as op:

    @op.auto
    def _min(x):
        return x.min()


with PolarsImpl.op(ops.Max()) as op:

    @op.auto
    def _max(x):
        return x.max()


with PolarsImpl.op(ops.Sum()) as op:

    @op.auto
    def _sum(x):
        return x.sum()


with PolarsImpl.op(ops.All()) as op:

    @op.auto
    def _all(x):
        return x.all()


with PolarsImpl.op(ops.Any()) as op:

    @op.auto
    def _any(x):
        return x.any()


with PolarsImpl.op(ops.IsNull()) as op:

    @op.auto
    def _is_null(x):
        return x.is_null()


with PolarsImpl.op(ops.IsNotNull()) as op:

    @op.auto
    def _is_not_null(x):
        return x.is_not_null()


with PolarsImpl.op(ops.FillNull()) as op:

    @op.auto
    def _fill_null(x, y):
        return x.fill_null(y)


with PolarsImpl.op(ops.DtYear()) as op:

    @op.auto
    def _dt_year(x):
        return x.dt.year().cast(pl.Int64)


with PolarsImpl.op(ops.DtMonth()) as op:

    @op.auto
    def _dt_month(x):
        return x.dt.month().cast(pl.Int64)


with PolarsImpl.op(ops.DtDay()) as op:

    @op.auto
    def _dt_day(x):
        return x.dt.day().cast(pl.Int64)


with PolarsImpl.op(ops.DtHour()) as op:

    @op.auto
    def _dt_hour(x):
        return x.dt.hour().cast(pl.Int64)


with PolarsImpl.op(ops.DtMinute()) as op:

    @op.auto
    def _dt_minute(x):
        return x.dt.minute().cast(pl.Int64)


with PolarsImpl.op(ops.DtSecond()) as op:

    @op.auto
    def _dt_second(x):
        return x.dt.second().cast(pl.Int64)


with PolarsImpl.op(ops.DtMillisecond()) as op:

    @op.auto
    def _dt_millisecond(x):
        return x.dt.millisecond().cast(pl.Int64)


with PolarsImpl.op(ops.DtDayOfWeek()) as op:

    @op.auto
    def _dt_day_of_week(x):
        return x.dt.weekday().cast(pl.Int64)


with PolarsImpl.op(ops.DtDayOfYear()) as op:

    @op.auto
    def _dt_day_of_year(x):
        return x.dt.ordinal_day().cast(pl.Int64)


with PolarsImpl.op(ops.DtDays()) as op:

    @op.auto
    def _days(x):
        return x.dt.total_days()


with PolarsImpl.op(ops.DtHours()) as op:

    @op.auto
    def _hours(x):
        return x.dt.total_hours()


with PolarsImpl.op(ops.DtMinutes()) as op:

    @op.auto
    def _minutes(x):
        return x.dt.total_minutes()


with PolarsImpl.op(ops.DtSeconds()) as op:

    @op.auto
    def _seconds(x):
        return x.dt.total_seconds()


with PolarsImpl.op(ops.DtMilliseconds()) as op:

    @op.auto
    def _milliseconds(x):
        return x.dt.total_milliseconds()


with PolarsImpl.op(ops.Sub()) as op:

    @op.extension(ops.DtSub)
    def _dt_sub(lhs, rhs):
        return lhs - rhs


with PolarsImpl.op(ops.RSub()) as op:

    @op.extension(ops.DtRSub)
    def _dt_rsub(rhs, lhs):
        return lhs - rhs


with PolarsImpl.op(ops.Add()) as op:

    @op.extension(ops.DtDurAdd)
    def _dt_dur_add(lhs, rhs):
        return lhs + rhs


with PolarsImpl.op(ops.RAdd()) as op:

    @op.extension(ops.DtDurRAdd)
    def _dt_dur_radd(rhs, lhs):
        return lhs + rhs


with PolarsImpl.op(ops.RowNumber()) as op:

    @op.auto
    def _row_number():
        return pl.int_range(start=1, end=pl.len() + 1, dtype=pl.Int64)


with PolarsImpl.op(ops.Rank()) as op:

    @op.auto
    def _rank(x):
        return x.rank("min").cast(pl.Int64)


with PolarsImpl.op(ops.DenseRank()) as op:

    @op.auto
    def _dense_rank(x):
        return x.rank("dense").cast(pl.Int64)


with PolarsImpl.op(ops.Shift()) as op:

    @op.auto
    def _shift(x, n, fill_value=None):
        return x.shift(n, fill_value=fill_value)


with PolarsImpl.op(ops.IsIn()) as op:

    @op.auto
    def _isin(x, *values):
        return pl.any_horizontal(
            (x == v if v is not None else x.is_null()) for v in values
        )


with PolarsImpl.op(ops.StrContains()) as op:

    @op.auto
    def _contains(x, y):
        return x.str.contains(y)


with PolarsImpl.op(ops.StrStartsWith()) as op:

    @op.auto
    def _starts_with(x, y):
        return x.str.starts_with(y)


with PolarsImpl.op(ops.StrEndsWith()) as op:

    @op.auto
    def _ends_with(x, y):
        return x.str.ends_with(y)


with PolarsImpl.op(ops.StrToLower()) as op:

    @op.auto
    def _lower(x):
        return x.str.to_lowercase()


with PolarsImpl.op(ops.StrToUpper()) as op:

    @op.auto
    def _upper(x):
        return x.str.to_uppercase()


with PolarsImpl.op(ops.StrReplaceAll()) as op:

    @op.auto
    def _replace_all(x, to_replace, replacement):
        return x.str.replace_all(to_replace, replacement)


with PolarsImpl.op(ops.StrLen()) as op:

    @op.auto
    def _string_length(x):
        return x.str.len_chars().cast(pl.Int64)


with PolarsImpl.op(ops.StrStrip()) as op:

    @op.auto
    def _str_strip(x):
        return x.str.strip_chars()


with PolarsImpl.op(ops.StrSlice()) as op:

    @op.auto
    def _str_slice(x, offset, length):
        return x.str.slice(offset, length)


with PolarsImpl.op(ops.Count()) as op:

    @op.auto
    def _count(x=None):
        return pl.len().cast(pl.Int64) if x is None else x.count().cast(pl.Int64)


with PolarsImpl.op(ops.Greatest()) as op:

    @op.auto
    def _greatest(*x):
        return pl.max_horizontal(*x)


with PolarsImpl.op(ops.Least()) as op:

    @op.auto
    def _least(*x):
        return pl.min_horizontal(*x)


with PolarsImpl.op(ops.Round()) as op:

    @op.auto
    def _round(x, digits=0):
        return x.round(digits)


with PolarsImpl.op(ops.Exp()) as op:

    @op.auto
    def _exp(x):
        return x.exp()


with PolarsImpl.op(ops.Log()) as op:

    @op.auto
    def _log(x):
        return x.log()


with PolarsImpl.op(ops.Floor()) as op:

    @op.auto
    def _floor(x):
        return x.floor()


with PolarsImpl.op(ops.Ceil()) as op:

    @op.auto
    def _ceil(x):
        return x.ceil()


with PolarsImpl.op(ops.StrToDateTime()) as op:

    @op.auto
    def _str_to_datetime(x):
        return x.str.to_datetime()


with PolarsImpl.op(ops.StrToDate()) as op:

    @op.auto
    def _str_to_date(x):
        return x.str.to_date()


with PolarsImpl.op(ops.FloorDiv()) as op:

    @op.auto
    def _floordiv(lhs, rhs):
        result_sign = (lhs < 0) ^ (rhs < 0)
        return (abs(lhs) // abs(rhs)) * pl.when(result_sign).then(-1).otherwise(1)
        # TODO: test some alternatives if this is too slow


with PolarsImpl.op(ops.Mod()) as op:

    @op.auto
    def _mod(lhs, rhs):
        return lhs % (abs(rhs) * pl.when(lhs >= 0).then(1).otherwise(-1))
        # TODO: see whether the following is faster:
        # pl.when(lhs >= 0).then(lhs % abs(rhs)).otherwise(lhs % -abs(rhs))


with PolarsImpl.op(ops.IsInf()) as op:

    @op.auto
    def _is_inf(x):
        return x.is_infinite()


with PolarsImpl.op(ops.IsNotInf()) as op:

    @op.auto
    def _is_not_inf(x):
        return x.is_finite()


with PolarsImpl.op(ops.IsNan()) as op:

    @op.auto
    def _is_nan(x):
        return x.is_nan()


with PolarsImpl.op(ops.IsNotNan()) as op:

    @op.auto
    def _is_not_nan(x):
        return x.is_not_nan()
