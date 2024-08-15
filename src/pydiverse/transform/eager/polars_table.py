from __future__ import annotations

import itertools
import uuid
from typing import Callable

import polars as pl

from pydiverse.transform.core import dtypes
from pydiverse.transform.core.expressions.expressions import Column, FunctionCall
from pydiverse.transform.core.expressions.symbolic_expressions import SymbolicExpression
from pydiverse.transform.core.expressions.translator import (
    Translator,
    TypedValue,
)
from pydiverse.transform.core.table_impl import AbstractTableImpl
from pydiverse.transform.errors import ExpressionError


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
            right.df, how=how, left_on=left_on, right_on=right_on, validate=validate
        )

    class ExpressionCompiler(
        AbstractTableImpl.ExpressionCompiler[
            "PolarsEager", TypedValue[Callable[[], pl.Expr]]
        ]
    ):
        def _translate_col(self, expr, **kwargs):
            def value():
                return pl.col(self.backend.underlying_col_name[expr.uuid])

            return TypedValue(value, expr.dtype)


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
    pass
