from __future__ import annotations

import dataclasses
import datetime
from typing import Any

import polars as pl

from pydiverse.transform import ops
from pydiverse.transform.backend.table_impl import TableImpl
from pydiverse.transform.backend.targets import Polars, Target
from pydiverse.transform.ops.core import OPType
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

    def col_type(self, col_name: str) -> dtypes.DType:
        return polars_type_to_pdt(self.df.schema[col_name])

    @staticmethod
    def compile_table_expr(expr: TableExpr) -> PolarsImpl:
        lf, context = table_expr_compile_with_context(expr)
        return PolarsImpl(lf.select(context.selects))

    @staticmethod
    def build_query(expr: TableExpr) -> str | None:
        return None

    @staticmethod
    def backend_marker() -> Target:
        return Polars(lazy=True)

    def export(self, target: Target) -> Any:
        if isinstance(target, Polars):
            return self.df if target.lazy else self.df.collect()

    def cols(self) -> list[str]:
        return self.df.columns

    def schema(self) -> dict[str, dtypes.DType]:
        return {
            name: polars_type_to_pdt(dtype) for name, dtype in self.df.schema.items()
        }


def col_expr_compile(expr: ColExpr, group_by: list[pl.Expr]) -> pl.Expr:
    assert not isinstance(expr, Col)
    if isinstance(expr, ColName):
        return pl.col(expr.name)

    elif isinstance(expr, ColFn):
        op = PolarsImpl.operator_registry.get_operator(expr.name)
        args: list[pl.Expr] = [col_expr_compile(arg, group_by) for arg in expr.args]
        impl = PolarsImpl.operator_registry.get_implementation(
            expr.name,
            tuple(arg.dtype for arg in expr.args),
        )

        # the `partition_by=` grouping overrides the `group_by` grouping
        partition_by = expr.context_kwargs.get("partition_by")
        if partition_by is None:
            partition_by = group_by

        arrange = expr.context_kwargs.get("arrange")

        if arrange:
            order_by, descending, nulls_last = zip(
                compile_order(order, group_by) for order in arrange
            )

        filter_cond = expr.context_kwargs.get("filter")

        if (
            op.ftype in (OPType.WINDOW, OPType.AGGREGATE)
            and arrange
            and not partition_by
        ):
            # order the args. if the table is grouped by group_by or
            # partition_by=, the groups will be sorted via over(order_by=)
            # anyways so it need not be done here.

            args = [
                arg.sort_by(by=order_by, descending=descending, nulls_last=nulls_last)
                for arg in args
            ]

        if op.ftype in (OPType.WINDOW, OPType.AGGREGATE) and filter_cond:
            # filtering needs to be done before applying the operator.
            args = [
                arg.filter(filter_cond) if isinstance(arg, pl.Expr) else arg
                for arg in args
            ]

        # if op.name in ("rank", "dense_rank"):
        #     assert len(args) == 0
        #     args = [pl.struct(merge_desc_nulls_last(ordering))]
        #     ordering = None

        value: pl.Expr = impl(*[arg for arg in args])

        if op.ftype == OPType.AGGREGATE:
            if filter_cond:
                # TODO: allow AGGRRGATE + `filter` context_kwarg
                raise NotImplementedError

            if partition_by:
                # technically, it probably wouldn't be too hard to support this in
                # polars.
                raise NotImplementedError

        # TODO: in the grouping / filter expressions, we should probably call
        # validate_table_args. look what it does and use it.
        # TODO: what happens if I put None or similar in a filter / partition_by?
        if op.ftype == OPType.WINDOW:
            # if `verb` != "muatate", we should give a warning that this only works
            # for polars

            if partition_by:
                # when doing sort_by -> over in polars, for whatever reason the
                # `nulls_last` argument is ignored. thus when both a grouping and an
                # arrangment are specified, we manually add the descending and
                # nulls_last markers to the ordering.
                order_by = None
                # if arrange:
                #     order_by = merge_desc_nulls_last(by, )
                value = value.over(partition_by, order_by=order_by)

            elif arrange:
                if op.ftype == OPType.AGGREGATE:
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
        raise NotImplementedError

    elif isinstance(expr, LiteralCol):
        return pl.lit(expr.val, dtype=pdt_type_to_polars(expr.dtype))

    else:
        raise AssertionError


# merges descending and null_last markers into the ordering expression
def merge_desc_nulls_last(self, order_exprs: list[Order]) -> list[pl.Expr]:
    with_signs: list[pl.Expr] = []
    for expr in order_exprs:
        numeric = col_expr_compile(expr.order_by, []).rank("dense").cast(pl.Int64)
        with_signs.append(-numeric if expr.descending else numeric)
    return [
        x.fill_null(
            pl.len().cast(pl.Int64) + 1
            if o.nulls_last
            else -(pl.len().cast(pl.Int64) + 1)
        )
        for x, o in zip(with_signs, order_exprs)
    ]


def compile_order(order: Order, group_by: list[pl.Expr]) -> tuple[pl.Expr, bool, bool]:
    return (
        col_expr_compile(order.order_by, group_by),
        order.descending,
        order.nulls_last,
    )


def compile_join_cond(expr: ColExpr) -> list[tuple[pl.Expr, pl.Expr]]:
    if isinstance(expr, ColFn):
        if expr.name == "__and__":
            return compile_join_cond(expr.args[0]) + compile_join_cond(expr.args[1])
        if expr.name == "__eq__":
            return (
                col_expr_compile(expr.args[0], []),
                col_expr_compile(expr.args[1], []),
            )

    raise AssertionError()


@dataclasses.dataclass
class CompilationContext:
    group_by: list[str]
    selects: list[str]

    def compiled_group_by(self) -> list[pl.Expr]:
        return [pl.col(name) for name in self.group_by]


def table_expr_compile_with_context(
    expr: TableExpr,
) -> tuple[pl.LazyFrame, CompilationContext]:
    if isinstance(expr, verbs.Alias):
        df, context = table_expr_compile_with_context(expr.table)
        setattr(df, expr.new_name)
        return df, context

    elif isinstance(expr, verbs.Select):
        df, context = table_expr_compile_with_context(expr.table)
        context.selects = [
            col
            for col in context.selects
            if col in set(col.name for col in expr.selects)
        ]
        return df, context

    elif isinstance(expr, verbs.Mutate):
        df, context = table_expr_compile_with_context(expr.table)
        context.selects.extend(
            name for name in expr.names if name not in set(context.selects)
        )
        return df.with_columns(
            **{
                name: col_expr_compile(
                    value,
                    context.compiled_group_by(),
                )
                for name, value in zip(expr.names, expr.values)
            }
        ), context

    elif isinstance(expr, verbs.Rename):
        df, context = table_expr_compile_with_context(expr.table)
        return df.rename(expr.name_map), context

    elif isinstance(expr, verbs.Join):
        left_df, left_context = table_expr_compile_with_context(expr.left)
        right_df, right_context = table_expr_compile_with_context(expr.right)
        assert not left_context.compiled_group_by
        assert not right_context.compiled_group_by
        left_on, right_on = zip(*compile_join_cond(expr.on))
        return left_df.join(
            right_df,
            left_on=left_on,
            right_on=right_on,
            how=expr.how,
            validate=expr.validate,
            suffix=expr.suffix,
        ), CompilationContext(
            [],
            left_context.selects
            + [col_name + expr.suffix for col_name in right_context.selects],
        )

    elif isinstance(expr, verbs.Filter):
        df, context = table_expr_compile_with_context(expr.table)
        return df.filter(
            col_expr_compile(expr.filters, context.compiled_group_by())
        ), context

    elif isinstance(expr, verbs.Arrange):
        df, context = table_expr_compile_with_context(expr.table)
        return df.sort(
            [
                compile_order(order, context.compiled_group_by())
                for order in expr.order_by
            ]
        ), context

    elif isinstance(expr, verbs.GroupBy):
        df, context = table_expr_compile_with_context(expr.table)
        return df, CompilationContext(
            (
                context.group_by + [col.name for col in expr.group_by]
                if expr.add
                else expr.group_by
            ),
            context.selects,
        )

    elif isinstance(expr, verbs.Ungroup):
        df, context = table_expr_compile_with_context(expr.table)
        return df, context

    elif isinstance(expr, verbs.Summarise):
        df, context = table_expr_compile_with_context(expr.table)
        compiled_group_by = context.compiled_group_by()
        return df.group_by(compiled_group_by).agg(
            **{
                name: col_expr_compile(
                    value,
                    compiled_group_by,
                )
                for name, value in zip(expr.names, expr.values)
            }
        ), CompilationContext([], context.group_by + expr.names)

    elif isinstance(expr, verbs.SliceHead):
        df, context = table_expr_compile_with_context(expr.table)
        assert len(context.group_by) == 0
        return df, context

    elif isinstance(expr, Table):
        assert isinstance(expr._impl, PolarsImpl)
        return expr._impl.df, CompilationContext([], expr.col_names())

    raise AssertionError


def polars_type_to_pdt(t: pl.DataType) -> dtypes.DType:
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

    raise TypeError(f"polars type {t} is not supported")


def pdt_type_to_polars(t: dtypes.DType) -> pl.DataType:
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

    raise TypeError(f"pydiverse.transform type {t} not supported for polars")


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

    raise TypeError(f"pydiverse.transform does not support python builtin type {t}")


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
        return pl.any_horizontal(x == v for v in values)


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
