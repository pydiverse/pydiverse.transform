from __future__ import annotations

import datetime

import polars as pl

from pydiverse.transform import ops
from pydiverse.transform.core import dtypes, verbs
from pydiverse.transform.core.col_expr import (
    CaseExpr,
    Col,
    ColExpr,
    ColFn,
    ColName,
    Order,
)
from pydiverse.transform.core.table_impl import TableImpl
from pydiverse.transform.core.util import OrderingDescriptor
from pydiverse.transform.core.verbs import TableExpr
from pydiverse.transform.ops.core import OPType


class PolarsEager(TableImpl):
    def __init__(self, name: str, df: pl.DataFrame):
        self.df = df
        super().__init__(name)

    # merges descending and null_last markers into the ordering expression
    def _merge_desc_nulls_last(
        self, ordering: list[OrderingDescriptor]
    ) -> list[pl.Expr]:
        with_signs = []
        for o in ordering:
            numeric = self.compiler.translate(o.order).rank("dense").cast(pl.Int64)
            with_signs.append(numeric if o.asc else -numeric)
        return [
            x.fill_null(
                -(pl.len().cast(pl.Int64) + 1)
                if o.nulls_first
                else pl.len().cast(pl.Int64) + 1
            )
            for x, o in zip(with_signs, ordering)
        ]


def compile_col_expr(expr: ColExpr, group_by: list[pl.Expr]) -> pl.Expr:
    assert not isinstance(expr, Col)
    if isinstance(expr, ColName):
        return pl.col(expr.name)

    elif isinstance(expr, ColFn):
        op = PolarsEager.operator_registry.get_operator(expr.name)
        args: list[pl.Expr] = [compile_col_expr(arg, group_by) for arg in expr.args]
        impl = PolarsEager.operator_registry.get_implementation(
            expr.name, tuple(arg._type for arg in expr.args)
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

    else:
        return pl.lit(expr, dtype=python_type_to_polars(type(expr)))


def compile_order(order: Order, group_by: list[pl.Expr]) -> tuple[pl.Expr, bool, bool]:
    return (
        compile_col_expr(order.order_by, group_by),
        order.descending,
        order.nulls_last,
    )


def compile_table_expr(expr: TableExpr) -> pl.LazyFrame:
    lf, _ = compile_table_expr_with_group_by(expr)
    return lf


def compile_join_cond(expr: ColExpr) -> list[tuple[pl.Expr, pl.Expr]]:
    if isinstance(expr, ColFn):
        if expr.name == "__and__":
            return compile_join_cond(expr.args[0]) + compile_join_cond(expr.args[1])
        if expr.name == "__eq__":
            return (
                compile_col_expr(expr.args[0], []),
                compile_col_expr(expr.args[1], []),
            )

    raise AssertionError()


def compile_table_expr_with_group_by(
    expr: TableExpr,
) -> tuple[pl.LazyFrame, list[pl.Expr]]:
    if isinstance(expr, verbs.Alias):
        table, group_by = compile_table_expr_with_group_by(expr.table)
        setattr(table, expr.new_name)
        return table, group_by

    elif isinstance(expr, verbs.Select):
        table, group_by = compile_table_expr_with_group_by(expr.table)
        return table.select(col.name for col in expr.selects), group_by

    elif isinstance(expr, verbs.Mutate):
        table, group_by = compile_table_expr_with_group_by(expr.table)
        return table.with_columns(
            **{
                name: compile_col_expr(
                    value,
                    group_by,
                )
                for name, value in zip(expr.names, expr.values)
            }
        ), group_by

    elif isinstance(expr, verbs.Rename):
        table, group_by = compile_table_expr_with_group_by(expr.table)
        return table.rename(expr.name_map), group_by

    elif isinstance(expr, verbs.Join):
        left, _ = compile_table_expr_with_group_by(expr.left)
        right, _ = compile_table_expr_with_group_by(expr.right)
        left_on, right_on = zip(*compile_join_cond(expr.on))
        return left.join(
            right,
            left_on=left_on,
            right_on=right_on,
            how=expr.how,
            validate=expr.validate,
            suffix=expr.suffix,
        ), []

    elif isinstance(expr, verbs.Filter):
        table, group_by = compile_table_expr_with_group_by(expr.table)
        return table.filter(compile_col_expr(expr.filters, group_by)), group_by

    elif isinstance(expr, verbs.Arrange):
        table, group_by = compile_table_expr_with_group_by(expr.table)
        return table.sort(
            [compile_order(order, group_by) for order in expr.order_by]
        ), group_by

    elif isinstance(expr, verbs.GroupBy):
        table, group_by = compile_table_expr_with_group_by(expr.table)
        new_group_by = compile_col_expr(expr.group_by, group_by)
        return table, (group_by + new_group_by) if expr.add else new_group_by

    elif isinstance(expr, verbs.Ungroup):
        table, _ = compile_table_expr_with_group_by(expr.table)
        return table, []

    elif isinstance(expr, verbs.SliceHead):
        table, group_by = compile_table_expr_with_group_by(expr.table)
        assert len(group_by) == 0
        return table, []

    raise AssertionError


def pdt_type_to_polars(t: pl.DataType) -> dtypes.DType:
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


def polars_type_to_pdt(t: dtypes.DType) -> pl.DataType:
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


with PolarsEager.op(ops.Mean()) as op:

    @op.auto
    def _mean(x):
        return x.mean()


with PolarsEager.op(ops.Min()) as op:

    @op.auto
    def _min(x):
        return x.min()


with PolarsEager.op(ops.Max()) as op:

    @op.auto
    def _max(x):
        return x.max()


with PolarsEager.op(ops.Sum()) as op:

    @op.auto
    def _sum(x):
        return x.sum()


with PolarsEager.op(ops.All()) as op:

    @op.auto
    def _all(x):
        return x.all()


with PolarsEager.op(ops.Any()) as op:

    @op.auto
    def _any(x):
        return x.any()


with PolarsEager.op(ops.IsNull()) as op:

    @op.auto
    def _is_null(x):
        return x.is_null()


with PolarsEager.op(ops.IsNotNull()) as op:

    @op.auto
    def _is_not_null(x):
        return x.is_not_null()


with PolarsEager.op(ops.FillNull()) as op:

    @op.auto
    def _fill_null(x, y):
        return x.fill_null(y)


with PolarsEager.op(ops.DtYear()) as op:

    @op.auto
    def _dt_year(x):
        return x.dt.year()


with PolarsEager.op(ops.DtMonth()) as op:

    @op.auto
    def _dt_month(x):
        return x.dt.month()


with PolarsEager.op(ops.DtDay()) as op:

    @op.auto
    def _dt_day(x):
        return x.dt.day()


with PolarsEager.op(ops.DtHour()) as op:

    @op.auto
    def _dt_hour(x):
        return x.dt.hour()


with PolarsEager.op(ops.DtMinute()) as op:

    @op.auto
    def _dt_minute(x):
        return x.dt.minute()


with PolarsEager.op(ops.DtSecond()) as op:

    @op.auto
    def _dt_second(x):
        return x.dt.second()


with PolarsEager.op(ops.DtMillisecond()) as op:

    @op.auto
    def _dt_millisecond(x):
        return x.dt.millisecond()


with PolarsEager.op(ops.DtDayOfWeek()) as op:

    @op.auto
    def _dt_day_of_week(x):
        return x.dt.weekday()


with PolarsEager.op(ops.DtDayOfYear()) as op:

    @op.auto
    def _dt_day_of_year(x):
        return x.dt.ordinal_day()


with PolarsEager.op(ops.DtDays()) as op:

    @op.auto
    def _days(x):
        return x.dt.total_days()


with PolarsEager.op(ops.DtHours()) as op:

    @op.auto
    def _hours(x):
        return x.dt.total_hours()


with PolarsEager.op(ops.DtMinutes()) as op:

    @op.auto
    def _minutes(x):
        return x.dt.total_minutes()


with PolarsEager.op(ops.DtSeconds()) as op:

    @op.auto
    def _seconds(x):
        return x.dt.total_seconds()


with PolarsEager.op(ops.DtMilliseconds()) as op:

    @op.auto
    def _milliseconds(x):
        return x.dt.total_milliseconds()


with PolarsEager.op(ops.Sub()) as op:

    @op.extension(ops.DtSub)
    def _dt_sub(lhs, rhs):
        return lhs - rhs


with PolarsEager.op(ops.RSub()) as op:

    @op.extension(ops.DtRSub)
    def _dt_rsub(rhs, lhs):
        return lhs - rhs


with PolarsEager.op(ops.Add()) as op:

    @op.extension(ops.DtDurAdd)
    def _dt_dur_add(lhs, rhs):
        return lhs + rhs


with PolarsEager.op(ops.RAdd()) as op:

    @op.extension(ops.DtDurRAdd)
    def _dt_dur_radd(rhs, lhs):
        return lhs + rhs


with PolarsEager.op(ops.RowNumber()) as op:

    @op.auto
    def _row_number():
        return pl.int_range(start=1, end=pl.len() + 1, dtype=pl.Int64)


with PolarsEager.op(ops.Rank()) as op:

    @op.auto
    def _rank(x):
        return x.rank("min").cast(pl.Int64)


with PolarsEager.op(ops.DenseRank()) as op:

    @op.auto
    def _dense_rank(x):
        return x.rank("dense").cast(pl.Int64)


with PolarsEager.op(ops.Shift()) as op:

    @op.auto
    def _shift(x, n, fill_value=None):
        return x.shift(n, fill_value=fill_value)


with PolarsEager.op(ops.IsIn()) as op:

    @op.auto
    def _isin(x, *values):
        return pl.any_horizontal(x == v for v in values)


with PolarsEager.op(ops.StrContains()) as op:

    @op.auto
    def _contains(x, y):
        return x.str.contains(y)


with PolarsEager.op(ops.StrStartsWith()) as op:

    @op.auto
    def _starts_with(x, y):
        return x.str.starts_with(y)


with PolarsEager.op(ops.StrEndsWith()) as op:

    @op.auto
    def _ends_with(x, y):
        return x.str.ends_with(y)


with PolarsEager.op(ops.StrToLower()) as op:

    @op.auto
    def _lower(x):
        return x.str.to_lowercase()


with PolarsEager.op(ops.StrToUpper()) as op:

    @op.auto
    def _upper(x):
        return x.str.to_uppercase()


with PolarsEager.op(ops.StrReplaceAll()) as op:

    @op.auto
    def _replace_all(x, to_replace, replacement):
        return x.str.replace_all(to_replace, replacement)


with PolarsEager.op(ops.StrLen()) as op:

    @op.auto
    def _string_length(x):
        return x.str.len_chars().cast(pl.Int64)


with PolarsEager.op(ops.StrStrip()) as op:

    @op.auto
    def _str_strip(x):
        return x.str.strip_chars()


with PolarsEager.op(ops.StrSlice()) as op:

    @op.auto
    def _str_slice(x, offset, length):
        return x.str.slice(offset, length)


with PolarsEager.op(ops.Count()) as op:

    @op.auto
    def _count(x=None):
        return pl.len() if x is None else x.count()


with PolarsEager.op(ops.Greatest()) as op:

    @op.auto
    def _greatest(*x):
        return pl.max_horizontal(*x)


with PolarsEager.op(ops.Least()) as op:

    @op.auto
    def _least(*x):
        return pl.min_horizontal(*x)
