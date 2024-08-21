from __future__ import annotations

import functools
import itertools
import operator
import uuid
from typing import Any, Callable

import polars as pl

from pydiverse.transform import ops
from pydiverse.transform.core import dtypes
from pydiverse.transform.core.expressions.expressions import (
    BaseExpression,
    CaseExpression,
    Column,
    FunctionCall,
    LiteralColumn,
)
from pydiverse.transform.core.expressions.symbolic_expressions import SymbolicExpression
from pydiverse.transform.core.expressions.translator import (
    Translator,
    TypedValue,
)
from pydiverse.transform.core.registry import TypedOperatorImpl
from pydiverse.transform.core.table_impl import AbstractTableImpl
from pydiverse.transform.core.util import OrderingDescriptor
from pydiverse.transform.core.util.util import translate_ordering
from pydiverse.transform.errors import ExpressionError, FunctionTypeError
from pydiverse.transform.ops.core import OPType


class PolarsEager(AbstractTableImpl):
    def __init__(self, name: str, df: pl.DataFrame):
        self.df = df
        self.join_translator = JoinTranslator()

        cols = {
            col.name: Column(col.name, self, _pdt_dtype(col.dtype))
            for col in df.iter_columns()
        }
        self.underlying_col_name: dict[uuid.UUID, str] = {
            col.uuid: f"{name}_{col.name}_{col.uuid.int}" for col in cols.values()
        }
        self.df = self.df.rename(
            {col.name: self.underlying_col_name[col.uuid] for col in cols.values()}
        )
        super().__init__(name, cols)

    def mutate(self, **kwargs):
        uuid_to_kwarg: dict[uuid.UUID, (str, BaseExpression)] = {
            self.named_cols.fwd[k]: (k, v) for (k, v) in kwargs.items()
        }
        self.underlying_col_name.update(
            {
                uuid: f"{self.name}_{col_name}_mut_{uuid.int}"
                for uuid, (col_name, _) in uuid_to_kwarg.items()
            }
        )

        polars_exprs = [
            self.cols[uuid].compiled().alias(self.underlying_col_name[uuid])
            for uuid in uuid_to_kwarg.keys()
        ]
        self.df = self.df.with_columns(*polars_exprs)

    def join(
        self,
        right: PolarsEager,
        on: SymbolicExpression,
        how: str,
        *,
        validate="m:m",
    ):
        # get the columns on which the data frames are joined
        left_on: list[str] = []
        right_on: list[str] = []
        for col1, col2 in self.join_translator.translate(on):
            if col2.uuid in self.cols and col1.uuid in right.cols:
                col1, col2 = col2, col1
            assert col1.uuid in self.cols and col2.uuid in right.cols
            left_on.append(self.underlying_col_name[col1.uuid])
            right_on.append(right.underlying_col_name[col2.uuid])

        self.underlying_col_name.update(right.underlying_col_name)

        self.df = self.df.join(
            right.df,
            how=how,
            left_on=left_on,
            right_on=right_on,
            validate=validate,
            coalesce=False,
        )

    def filter(self, *args: SymbolicExpression):
        if not args:
            return
        pl_expr, dtype = self.compiler.translate(functools.reduce(operator.and_, args))
        assert isinstance(dtype, dtypes.Bool)
        self.df = self.df.filter(pl_expr())

    def alias(self, new_name: str | None = None):
        new_name = new_name or self.name
        return self.__class__(new_name, self.export())

    def arrange(self, ordering: list[OrderingDescriptor]):
        self.df = self.df.sort(
            by=[self.compiler.translate(o.order).value() for o in ordering],
            nulls_last=[not o.nulls_first for o in ordering],
            descending=[not o.asc for o in ordering],
        )

    def summarise(self, **kwargs):
        uuid_to_kwarg: dict[uuid.UUID, (str, BaseExpression)] = {
            self.named_cols.fwd[k]: (k, v) for (k, v) in kwargs.items()
        }
        self.underlying_col_name.update(
            {
                uuid: f"{self.name}_{col_name}_summarise_{uuid.int}"
                for uuid, (col_name, _) in uuid_to_kwarg.items()
            }
        )

        agg_exprs: list[pl.Expr] = [
            self.cols[uuid].compiled().alias(self.underlying_col_name[uuid])
            for uuid in uuid_to_kwarg.keys()
        ]
        group_exprs: list[pl.Expr] = [
            pl.col(self.underlying_col_name[col.uuid]) for col in self.grouped_by
        ]

        if self.grouped_by:
            # retain the cols the table was grouped by and add the aggregation cols
            self.df = self.df.group_by(*group_exprs).agg(*agg_exprs)
        else:
            self.df = self.df.select(*agg_exprs)

    def export(self) -> pl.DataFrame:
        return self.df.select(
            **{
                name: self.underlying_col_name[uuid]
                for (name, uuid) in self.selected_cols()
            }
        )

    def is_aligned_with(self, col: Column | LiteralColumn) -> bool:
        if isinstance(col, Column):
            return (
                isinstance(col.table, type(self))
                and col.table.df.height == self.df.height
            )
        if isinstance(col, LiteralColumn):
            return issubclass(col.backend, type(self)) and (
                not isinstance(col.typed_value.value, pl.Series)
                or len(col.typed_value.value) == self.df.height
            )  # not a series => scalar

    class ExpressionCompiler(
        AbstractTableImpl.ExpressionCompiler[
            "PolarsEager", TypedValue[Callable[[], pl.Expr]]
        ]
    ):
        def _translate_col(self, expr, **kwargs):
            def value():
                return pl.col(self.backend.underlying_col_name[expr.uuid])

            return TypedValue(value, expr.dtype)

        def _translate_function(
            self,
            expr: FunctionCall,
            implementation: TypedOperatorImpl,
            op_args: list[TypedValue[Callable[[], pl.Expr]]],
            context_kwargs: dict[str, Any],
            *,
            verb: str | None = None,
            **kwargs,
        ) -> TypedValue[Callable[[], pl.Expr]]:
            pl_result_type = _pl_dtype(implementation.rtype)

            internal_kwargs = {}

            def value(**kw):
                return implementation(
                    *[arg.value(**kw) for arg in op_args],
                    _tbl=self.backend,
                    _result_type=pl_result_type,
                    **internal_kwargs,
                )

            op = implementation.operator

            if op.ftype == OPType.WINDOW:
                if verb != "mutate":
                    raise FunctionTypeError(
                        "window function are only allowed inside a mutate"
                    )

                if arrange := context_kwargs.get("arrange"):
                    ordering = translate_ordering(self.backend, arrange)
                    internal_kwargs["_ordering"] = ordering

                    # emulate that the function is computed on a table sorted by
                    # `ordering`. then restore the original order. this is equivalent
                    # to giving `pl.Expr.sort_by()` the permutation inverse to
                    # `ordering`.
                    def sorted_value(value):
                        inv_permutation = pl.int_range(
                            pl.len(), dtype=pl.Int64
                        ).sort_by(
                            by=[self.translate(o.order).value() for o in ordering],
                            nulls_last=[not o.nulls_first for o in ordering],
                            descending=[not o.asc for o in ordering],
                        )
                        return value().sort_by(inv_permutation)

                    # need to bind `value` inside `sorted_value` so that it refers to
                    # the original `value`.
                    value = functools.partial(sorted_value, value)

                if self.backend.grouped_by:

                    def partitioned_value(value):
                        group_exprs: list[pl.Expr] = [
                            pl.col(self.backend.underlying_col_name[col.uuid])
                            for col in self.backend.grouped_by
                        ]
                        return value().over(*group_exprs)

                    value = functools.partial(partitioned_value, value)

            return TypedValue(
                value,
                implementation.rtype,
                PolarsEager._get_op_ftype(
                    op_args,
                    op,
                    OPType.WINDOW
                    if op.ftype == OPType.AGGREGATE and verb == "mutate"
                    else None,
                ),
            )

        def _translate_case(
            self,
            expr: CaseExpression,
            switching_on: TypedValue[Callable[[], pl.Expr]] | None,
            cases: list[
                tuple[
                    TypedValue[Callable[[], pl.Expr]], TypedValue[Callable[[], pl.Expr]]
                ]
            ],
            default: TypedValue[Callable[[], pl.Expr]],
            **kwargs,
        ) -> TypedValue[Callable[[], pl.Expr]]:
            def value():
                if switching_on is not None:
                    switching_on_v = switching_on.value()
                    conds = [
                        match_expr.value() == switching_on_v for match_expr, _ in cases
                    ]
                else:
                    conds = [case[0].value() for case in cases]

                pl_expr = pl.when(conds[0]).then(cases[0][1].value())
                for cond, (_, value) in zip(conds[1:], cases[1:]):
                    pl_expr = pl_expr.when(cond).then(value.value())
                return pl_expr.otherwise(default.value())

            result_dtype, result_ftype = self._translate_case_common(
                expr, switching_on, cases, default, **kwargs
            )

            return TypedValue(value, result_dtype, result_ftype)

    class AlignedExpressionEvaluator(
        AbstractTableImpl.AlignedExpressionEvaluator[TypedValue[pl.Series]]
    ):
        def _translate_col(self, expr: Column, **kwargs) -> TypedValue[pl.Series]:
            return TypedValue(
                expr.table.df.get_column(expr.table.underlying_col_name[expr.uuid]),
                expr.table.cols[expr.uuid].dtype,
            )


class JoinTranslator(Translator[tuple]):
    """
    This translator takes a conjunction (AND) of equality checks and returns
    a tuple of tuple where the inner tuple contains the left and right column
    of the equality checks.
    """

    def _translate(self, expr, **kwargs):
        if isinstance(expr, Column):
            return expr
        if isinstance(expr, FunctionCall):
            if expr.name == "__eq__":
                c1 = expr.args[0]
                c2 = expr.args[1]
                assert isinstance(c1, Column) and isinstance(c2, Column)
                return ((c1, c2),)
            if expr.name == "__and__":
                return tuple(itertools.chain(*expr.args))
        raise ExpressionError(
            f"invalid ON clause element: {expr}. only a conjunction of equalities"
            " is supported"
        )


def _pdt_dtype(t: pl.DataType) -> dtypes.DType:
    if isinstance(t, pl.Float64):
        return dtypes.Float()
    elif isinstance(t, pl.Int64):
        return dtypes.Int()
    elif isinstance(t, pl.Boolean):
        return dtypes.Bool()
    elif isinstance(t, pl.String):
        return dtypes.String()
    elif isinstance(t, pl.Datetime):
        return dtypes.DateTime()

    raise TypeError(f"polars type {t} is not supported")


def _pl_dtype(t: dtypes.DType) -> pl.DataType:
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

    raise TypeError(f"pydiverse.transform type {t} not supported for polars")


with PolarsEager.op(ops.Mean()) as op:

    @op.auto
    def _mean(x):
        return x.mean()


with PolarsEager.op(ops.Min()) as op:

    @op.auto
    def _mean(x):
        return x.min()


with PolarsEager.op(ops.Max()) as op:

    @op.auto
    def _mean(x):
        return x.max()


with PolarsEager.op(ops.RowNumber()) as op:

    @op.auto
    def _row_number():
        return pl.int_range(start=1, end=pl.len() + 1, dtype=pl.Int64)
