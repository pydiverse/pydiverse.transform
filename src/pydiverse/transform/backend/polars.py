from __future__ import annotations

import datetime
from types import NoneType
from typing import Any

import polars as pl

from pydiverse.transform import ops
from pydiverse.transform.backend.table_impl import TableImpl
from pydiverse.transform.backend.targets import Polars, Target
from pydiverse.transform.ops.core import Ftype
from pydiverse.transform.pipe.table import Table
from pydiverse.transform.tree import dtypes, verbs
from pydiverse.transform.tree.col_expr import (
    CaseExpr,
    Col,
    ColExpr,
    ColFn,
    ColName,
    LiteralCol,
    Order,
)
from pydiverse.transform.tree.table_expr import TableExpr


class PolarsImpl(TableImpl):
    def __init__(self, df: pl.DataFrame | pl.LazyFrame):
        self.df = df if isinstance(df, pl.LazyFrame) else df.lazy()

    @staticmethod
    def build_query(expr: TableExpr) -> str | None:
        return None

    @staticmethod
    def export(expr: TableExpr, target: Target) -> Any:
        lf, select, _ = compile_table_expr(expr)
        lf = lf.select(select)
        if isinstance(target, Polars):
            return lf if target.lazy else lf.collect()

    def col_names(self) -> list[str]:
        return self.df.columns

    def schema(self) -> dict[str, dtypes.Dtype]:
        return {
            name: polars_type_to_pdt(dtype)
            for name, dtype in self.df.collect_schema().items()
        }

    def clone(self) -> PolarsImpl:
        return PolarsImpl(self.df.clone())


# merges descending and null_last markers into the ordering expression
def merge_desc_nulls_last(
    order_by: list[pl.Expr], descending: list[bool], nulls_last: list[bool]
) -> list[pl.Expr]:
    with_signs: list[pl.Expr] = []
    for ord, desc in zip(order_by, descending):
        numeric = ord.rank("dense").cast(pl.Int64)
        with_signs.append(-numeric if desc else numeric)
    return [
        expr.fill_null(
            pl.len().cast(pl.Int64) + 1 if nl else -(pl.len().cast(pl.Int64) + 1)
        )
        for expr, nl in zip(with_signs, nulls_last)
    ]


def compile_order(order: Order) -> tuple[pl.Expr, bool, bool]:
    return (
        compile_col_expr(order.order_by),
        order.descending,
        order.nulls_last,
    )


def compile_col_expr(expr: ColExpr) -> pl.Expr:
    assert not isinstance(expr, Col)
    if isinstance(expr, ColName):
        return pl.col(expr.name)

    elif isinstance(expr, ColFn):
        op = PolarsImpl.registry.get_op(expr.name)
        args: list[pl.Expr] = [compile_col_expr(arg) for arg in expr.args]
        impl = PolarsImpl.registry.get_impl(
            expr.name,
            tuple(arg.dtype for arg in expr.args),
        )

        partition_by = expr.context_kwargs.get("partition_by")
        if partition_by:
            partition_by = [compile_col_expr(col) for col in partition_by]

        arrange = expr.context_kwargs.get("arrange")
        if arrange:
            order_by, descending, nulls_last = zip(
                *[compile_order(order) for order in arrange]
            )

        filter_cond = expr.context_kwargs.get("filter")
        if filter_cond:
            filter_cond = [compile_col_expr(cond) for cond in filter_cond]

        # The following `if` block is absolutely unecessary and just an optimization.
        # Otherwise, `over` would be used for sorting, but we cannot pass descending /
        # nulls_last there and the required workaround is probably slower than polars`s
        # native `sort_by`.
        if arrange and not partition_by:
            # order the args. if the table is grouped by group_by or
            # partition_by=, the groups will be sorted via over(order_by=)
            # anyways so it need not be done here.
            args = [
                arg.sort_by(by=order_by, descending=descending, nulls_last=nulls_last)
                if isinstance(arg, pl.Expr)
                else arg
                for arg in args
            ]

        if filter_cond:
            # filtering needs to be done before applying the operator.
            args = [
                arg.filter(filter_cond) if isinstance(arg, pl.Expr) else arg
                for arg in args
            ]

        if op.name in ("rank", "dense_rank"):
            assert len(args) == 0
            args = [pl.struct(merge_desc_nulls_last(order_by, descending, nulls_last))]
            arrange = None

        value: pl.Expr = impl(*args)

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
            inv_permutation = pl.int_range(0, pl.len(), dtype=pl.Int64).sort_by(
                by=order_by,
                descending=descending,
                nulls_last=nulls_last,
            )
            value = value.sort_by(inv_permutation)

        return value

    elif isinstance(expr, CaseExpr):
        assert len(expr.cases) >= 1
        compiled = pl  # to initialize the when/then-chain
        for cond, val in expr.cases:
            compiled = compiled.when(compile_col_expr(cond)).then(compile_col_expr(val))
        return compiled.otherwise(compile_col_expr(expr.default_val))

    elif isinstance(expr, LiteralCol):
        if isinstance(expr.dtype, dtypes.String):
            return pl.lit(expr.val)  # polars interprets strings as column names
        return expr.val

    else:
        raise AssertionError


def compile_join_cond(expr: ColExpr) -> list[tuple[pl.Expr, pl.Expr]]:
    if isinstance(expr, ColFn):
        if expr.name == "__and__":
            return compile_join_cond(expr.args[0]) + compile_join_cond(expr.args[1])
        if expr.name == "__eq__":
            return [
                (
                    compile_col_expr(expr.args[0]),
                    compile_col_expr(expr.args[1]),
                )
            ]

    raise AssertionError()


# returns the compiled LazyFrame, the list of selected cols (selection on the frame
# must happen at the end since we need to store intermediate columns) and the cols
# the table is currently grouped by.
def compile_table_expr(
    expr: TableExpr,
) -> tuple[pl.LazyFrame, list[str], list[str]]:
    if isinstance(expr, verbs.Select):
        df, select, group_by = compile_table_expr(expr.table)
        select = [
            col for col in select if col in set(col.name for col in expr.selected)
        ]

    elif isinstance(expr, verbs.Drop):
        df, select, group_by = compile_table_expr(expr.table)
        select = [
            col for col in select if col not in set(col.name for col in expr.dropped)
        ]

    elif isinstance(expr, verbs.Rename):
        df, select, group_by = compile_table_expr(expr.table)
        df = df.rename(expr.name_map)
        select = [
            (expr.name_map[name] if name in expr.name_map else name) for name in select
        ]
        group_by = [
            (expr.name_map[name] if name in expr.name_map else name)
            for name in group_by
        ]

    elif isinstance(expr, verbs.Mutate):
        df, select, group_by = compile_table_expr(expr.table)
        select.extend(name for name in expr.names if name not in set(select))
        df = df.with_columns(
            **{
                name: compile_col_expr(value)
                for name, value in zip(expr.names, expr.values)
            }
        )

    elif isinstance(expr, verbs.Join):
        # may assume the tables were not grouped before join
        left_df, left_select, _ = compile_table_expr(expr.table)
        right_df, right_select, _ = compile_table_expr(expr.right)

        left_on, right_on = zip(*compile_join_cond(expr.on))
        # we want a suffix everywhere but polars only appends it to duplicate columns
        right_df = right_df.rename(
            {name: name + expr.suffix for name in right_df.columns}
        )

        df = left_df.join(
            right_df,
            left_on=left_on,
            right_on=right_on,
            how=expr.how,
            validate=expr.validate,
            coalesce=False,
        )
        select = left_select + [col_name + expr.suffix for col_name in right_select]
        group_by = []

    elif isinstance(expr, verbs.Filter):
        df, select, group_by = compile_table_expr(expr.table)
        if expr.filters:
            df = df.filter([compile_col_expr(fil) for fil in expr.filters])

    elif isinstance(expr, verbs.Arrange):
        df, select, group_by = compile_table_expr(expr.table)
        order_by, descending, nulls_last = zip(
            *[compile_order(order) for order in expr.order_by]
        )
        df = df.sort(
            order_by,
            descending=descending,
            nulls_last=nulls_last,
            maintain_order=True,
        )

    elif isinstance(expr, verbs.GroupBy):
        df, select, group_by = compile_table_expr(expr.table)
        group_by = (
            group_by + [col.name for col in expr.group_by]
            if expr.add
            else [col.name for col in expr.group_by]
        )

    elif isinstance(expr, verbs.Ungroup):
        df, select, group_by = compile_table_expr(expr.table)

    elif isinstance(expr, verbs.Summarise):
        df, select, group_by = compile_table_expr(expr.table)
        aggregations = [
            compile_col_expr(value).alias(name)
            for name, value in zip(expr.names, expr.values)
        ]

        if group_by:
            df = df.group_by(*(pl.col(name) for name in group_by)).agg(*aggregations)
        else:
            df = df.select(*aggregations)

        select = expr.names
        group_by = []

    elif isinstance(expr, verbs.SliceHead):
        df, select, group_by = compile_table_expr(expr.table)
        assert len(group_by) == 0
        df = df.slice(expr.offset, expr.n)

    elif isinstance(expr, Table):
        assert isinstance(expr._impl, PolarsImpl)
        df = expr._impl.df
        select = expr.col_names()
        group_by = []

    else:
        raise AssertionError

    return df, select, group_by


def polars_type_to_pdt(t: pl.DataType) -> dtypes.Dtype:
    if t.is_float():
        return dtypes.Float()
    elif t.is_integer():
        return dtypes.Int()
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
    if isinstance(t, dtypes.Float):
        return pl.Float64()
    elif isinstance(t, dtypes.Int):
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


def python_type_to_polars(t: type) -> pl.DataType:
    if t is int:
        return pl.Int64()
    elif t is float:
        return pl.Float64()
    elif t is bool:
        return pl.Boolean()
    elif t is str:
        return pl.String()
    elif t is datetime.datetime:
        return pl.Datetime()
    elif t is datetime.date:
        return pl.Date()
    elif t is datetime.timedelta:
        return pl.Duration()
    elif t is NoneType:
        return pl.Null()

    raise TypeError(f"python builtin type {t} is not supported by pydiverse.transform")


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
        return x.dt.year()


with PolarsImpl.op(ops.DtMonth()) as op:

    @op.auto
    def _dt_month(x):
        return x.dt.month()


with PolarsImpl.op(ops.DtDay()) as op:

    @op.auto
    def _dt_day(x):
        return x.dt.day()


with PolarsImpl.op(ops.DtHour()) as op:

    @op.auto
    def _dt_hour(x):
        return x.dt.hour()


with PolarsImpl.op(ops.DtMinute()) as op:

    @op.auto
    def _dt_minute(x):
        return x.dt.minute()


with PolarsImpl.op(ops.DtSecond()) as op:

    @op.auto
    def _dt_second(x):
        return x.dt.second()


with PolarsImpl.op(ops.DtMillisecond()) as op:

    @op.auto
    def _dt_millisecond(x):
        return x.dt.millisecond()


with PolarsImpl.op(ops.DtDayOfWeek()) as op:

    @op.auto
    def _dt_day_of_week(x):
        return x.dt.weekday()


with PolarsImpl.op(ops.DtDayOfYear()) as op:

    @op.auto
    def _dt_day_of_year(x):
        return x.dt.ordinal_day()


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
        return pl.len() if x is None else x.count()


with PolarsImpl.op(ops.Greatest()) as op:

    @op.auto
    def _greatest(*x):
        return pl.max_horizontal(*x)


with PolarsImpl.op(ops.Least()) as op:

    @op.auto
    def _least(*x):
        return pl.min_horizontal(*x)
