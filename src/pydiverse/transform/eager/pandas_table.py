from __future__ import annotations

import functools
import itertools
import operator as py_operator
import uuid
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
import pandas.core.dtypes.cast
import pandas.core.dtypes.dtypes

from pydiverse.transform import ops
from pydiverse.transform.core import dtypes
from pydiverse.transform.core.expressions import (
    Column,
    FunctionCall,
    LiteralColumn,
    SymbolicExpression,
)
from pydiverse.transform.core.expressions.translator import Translator, TypedValue
from pydiverse.transform.core.util import OrderingDescriptor, translate_ordering
from pydiverse.transform.eager.eager_table import EagerTableImpl, uuid_to_str
from pydiverse.transform.errors import (
    AlignmentError,
    ExpressionError,
    FunctionTypeError,
)
from pydiverse.transform.ops import OPType

__all__ = [
    "PandasTableImpl",
]


class PandasTableImpl(EagerTableImpl):
    """Pandas backend

    Attributes:
        df: The current, eager dataframe. It's columns get renamed to prevent
            name collisions, and it also contains all columns that have been
            computed at some point. This allows for a more lazy style API.
    """

    def __init__(self, name: str, df: pd.DataFrame):
        self.df = fast_pd_convert_dtypes(df)
        self.join_translator = self.JoinTranslator()

        columns = {
            name: Column(name=name, table=self, dtype=self._dtype_from_pd(dtype))
            for name, dtype in self.df.dtypes.items()
        }

        # Rename columns
        self.df_name_mapping = {
            col.uuid: f"{name}_{col.name}_" + uuid_to_str(col.uuid)
            for col in columns.values()
        }  # type: dict[uuid.UUID: str]

        self.df = self.df.rename(
            columns={
                name: self.df_name_mapping[col.uuid] for name, col in columns.items()
            }
        )

        super().__init__(name=name, columns=columns)

    def _dtype_from_pd(self, type_: Any) -> dtypes.DType:
        if pd.api.types.is_integer_dtype(type_):
            return dtypes.Int()
        if pd.api.types.is_float_dtype(type_):
            return dtypes.Float()
        if pd.api.types.is_bool_dtype(type_):
            return dtypes.Bool()
        if pd.api.types.is_string_dtype(type_):
            return dtypes.String()
        if pd.api.types.is_datetime64_any_dtype(type_):
            return dtypes.DateTime()

        raise NotImplementedError(f"Unsupported type: {type_}")

    def _dtype_to_pd(self, dtype: dtypes.DType) -> Any:
        if dtypes.Int().same_kind(dtype):
            return pd.Int64Dtype()
        if dtypes.Float().same_kind(dtype):
            return pd.Float64Dtype()
        if dtypes.Bool().same_kind(dtype):
            return pd.BooleanDtype()
        if dtypes.String().same_kind(dtype):
            return pd.StringDtype()
        if dtypes.DateTime().same_kind(dtype):
            return "datetime64[ns]"

        raise NotImplementedError(f"Invalid type: {dtype}")

    def is_aligned_with(self, col) -> bool:
        len_self = len(self.df.index)

        if isinstance(col, Column):
            if not isinstance(col.table, type(self)):
                return False
            return len(col.table.df.index) == len_self

        if isinstance(col, LiteralColumn):
            if not issubclass(col.backend, type(self)):
                return False
            if isinstance(col.typed_value.value, pd.Series):
                return len(col.typed_value.value) == len_self
            return True  # A constant value (eg int, float, str...)

        raise ValueError

    #### Verb Operations ####

    def alias(self, name=None):
        # Creating a new table object also acts like a garbage collecting mechanism.
        new_name = name or self.name
        return self.__class__(new_name, self.collect())

    def collect(self) -> pd.DataFrame:
        # SELECT -> Apply mask
        selected_cols_name_map = {
            self.df_name_mapping[uuid]: name for name, uuid in self.selected_cols()
        }
        masked_df = self.df[[*selected_cols_name_map.keys()]]

        # rename columns from internal naming scheme to external names
        result = masked_df.rename(columns=selected_cols_name_map)

        # Add metadata
        result.attrs["name"] = self.name
        return result

    def mutate(self, **kwargs):
        uuid_kwargs = {self.named_cols.fwd[k]: (k, v) for k, v in kwargs.items()}
        self.df_name_mapping.update(
            {
                uuid: f"{self.name}_mutate_{name}_" + uuid_to_str(uuid)
                for uuid, (name, _) in uuid_kwargs.items()
            }
        )

        # Update Dataframe
        grouper = self.df_grouper()
        cols = {
            self.df_name_mapping[uuid]: self.cols[uuid].compiled(
                self.df, grouper=grouper
            )
            for uuid in uuid_kwargs.keys()
        }

        self.df = self.df.copy(deep=False)
        for col_name, col in cols.items():
            self.df[col_name] = col
        self.df = fast_pd_convert_dtypes(self.df)

    def join(
        self,
        right: PandasTableImpl,
        on: SymbolicExpression,
        how: str,
        *,
        validate=None,
    ):
        """
        :param on: Symbolic expression consisting of anded column equality checks.
        """

        # Parse ON condition
        on_cols = []
        for col1, col2 in self.join_translator.translate(on):
            if col1.uuid in self.cols and col2.uuid in right.cols:
                on_cols.append((col1, col2))
            elif col2.uuid in self.cols and col1.uuid in right.cols:
                on_cols.append((col2, col1))
            else:
                raise Exception

        if validate is not None:
            pd_validate = {"1:?": "1:1"}[validate]
        else:
            pd_validate = None

        # Perform Join
        left_on, right_on = zip(*on_cols)
        left_on = [self.df_name_mapping[col.uuid] for col in left_on]
        right_on = [right.df_name_mapping[col.uuid] for col in right_on]

        self.df_name_mapping.update(right.df_name_mapping)

        # TODO: Implement more merge operations that preserve the index / ordering.
        if how == "left" and validate == "1:?":
            # Perform an order and index preserving join
            original_index = self.df.index
            tmp_df = self.df.reset_index(drop=True)
            merged_df = tmp_df.merge(
                right.df,
                how=how,
                left_on=left_on,
                right_on=right_on,
                validate=pd_validate,
            ).loc[tmp_df.index]
            merged_df.index = original_index
            self.df = merged_df
        else:
            self.df = self.df.merge(
                right.df,
                how=how,
                left_on=left_on,
                right_on=right_on,
                validate=pd_validate,
            )

    def filter(self, *args: SymbolicExpression):
        if not args:
            return

        compiled, dtype = self.compiler.translate(
            functools.reduce(py_operator.and_, args)
        )
        assert isinstance(dtype, dtypes.Bool)

        condition = compiled(self.df)
        self.df = self.df.loc[condition]

    def arrange(self, ordering):
        self.df = self.sort_df(self.df, ordering)

    def summarise(self, **kwargs):
        uuid_kwargs = {self.named_cols.fwd[k]: (k, v) for k, v in kwargs.items()}
        self.df_name_mapping.update(
            {
                uuid: f"{self.name}_summarise_{name}_" + uuid_to_str(uuid)
                for uuid, (name, _) in uuid_kwargs.items()
            }
        )

        # Update Dataframe
        grouper = self.grouped_df().grouper if self.grouped_by else None
        aggr_cols = {
            self.df_name_mapping[uuid]: self.cols[uuid].compiled(
                self.df, grouper=grouper
            )
            for uuid in uuid_kwargs.keys()
        }

        # Grouped Dataframe requires different operations compared to ungrouped df.
        if self.grouped_by:
            columns = {
                k: fast_pd_convert_dtypes(v.rename(k)) for k, v in aggr_cols.items()
            }
            self.df = pd.concat(columns, axis="columns", copy=False).reset_index()
        else:
            self.df = fast_pd_convert_dtypes(pd.DataFrame(aggr_cols, index=[0]))

    def slice_head(self, n: int, offset: int):
        self.df = self.df[offset : n + offset]

    #### EXPRESSIONS ####

    class ExpressionCompiler(
        EagerTableImpl.ExpressionCompiler[
            "PandasTableImpl", TypedValue[Callable[[pd.DataFrame], pd.Series]]
        ]
    ):
        def _translate_col(self, expr, **kwargs):
            df_col_name = self.backend.df_name_mapping[expr.uuid]

            def df_col(df, **kw):
                return df[df_col_name]

            return TypedValue(df_col, expr.dtype)

        def _translate_literal_col(self, expr, **kwargs):
            if not self.backend.is_aligned_with(expr):
                raise AlignmentError(
                    "Literal column isn't aligned with this table. "
                    f"Literal Column: {expr}"
                )

            def value(df, **kw):
                literal_value = expr.typed_value.value
                if isinstance(literal_value, pd.Series):
                    literal_len = len(literal_value.index)
                    if len(df.index) != literal_len:
                        raise AlignmentError(
                            "Literal column isn't aligned with this table. Make sure"
                            f" that they have the same length ({len(df.index)} vs"
                            f" {literal_len}). This might be the case if you are using"
                            " literal columns inside a window function.\nLiteral"
                            f" Column: {expr}"
                        )

                    series = literal_value.copy()
                    series.index = df.index
                    return series
                return literal_value

            return TypedValue(value, expr.typed_value.dtype, expr.typed_value.ftype)

        def _translate_function(
            self, expr, implementation, op_args, context_kwargs, *, verb=None, **kwargs
        ):
            internal_kwargs = {}

            def value(df, *, grouper=None, **kw):
                args = [arg.value(df, grouper=grouper, **kw) for arg in op_args]
                kwargs = {
                    "_tbl": self.backend,
                    "_df": df,
                    **internal_kwargs,
                }

                # Element wise operator
                # -> Perform on ungrouped columns
                if implementation.operator.ftype == OPType.EWISE:
                    return implementation(*args, **kwargs)

                # Get arguments; If there are no arguments, use dataframe index
                # instead for grouping.
                if len(args) > 0:
                    series, *args = args
                else:
                    series = df.index.to_series()

                def bound_impl(x):
                    """
                    Helper that binds the *args and **kwargs to the implementation.
                    Takes one argument - the rest is already supplied.
                    """
                    return implementation(x, *args, **kwargs)

                # Grouped
                if grouper is not None:
                    series = series.groupby(grouper)

                    if verb == "summarise":
                        if variant := implementation.get_variant("aggregate"):
                            return variant(series, *args, **kwargs)
                        return series.aggregate(bound_impl)
                    if variant := implementation.get_variant("transform"):
                        return variant(series, *args, **kwargs)
                    return series.transform(bound_impl)

                # Ungrouped
                scalar = bound_impl(series)
                if verb == "summarise":
                    return scalar

                pd_dtype = self.backend._dtype_to_pd(implementation.rtype)
                return pd.Series(scalar, index=series.index, dtype=pd_dtype)

            operator = implementation.operator

            if operator.ftype == OPType.WINDOW:
                if verb != "mutate":
                    raise FunctionTypeError(
                        "Window function are only allowed inside a mutate."
                    )

                if arrange := context_kwargs.get("arrange"):
                    ordering = translate_ordering(self.backend, arrange)
                    internal_kwargs["_ordering"] = ordering

                value = self.arranged_window(
                    value, operator, context_kwargs, verb=verb, **kwargs
                )

            override_ftype = (
                OPType.WINDOW
                if operator.ftype == OPType.AGGREGATE and verb == "mutate"
                else None
            )
            ftype = self.backend._get_op_ftype(op_args, operator, override_ftype)
            return TypedValue(value, implementation.rtype, ftype)

        def _translate_case(self, expr, switching_on, cases, default, **kwargs):
            def value(df, **kw):
                pd_dtype = self.backend._dtype_to_pd(result_dtype)
                grouper = kw.get("grouper")

                default_ = default.value(df, **kw)

                if not isinstance(default_, pd.Series):
                    result_index = df.index if grouper is None else grouper.result_index
                    result = pd.Series(default_, index=result_index, dtype=pd_dtype)
                else:
                    result = default_

                if switching_on is not None:
                    switching_on_ = switching_on.value(df, **kw)
                else:
                    switching_on_ = None

                for cond, val in cases[::-1]:
                    cond_ = cond.value(df, **kw)
                    val_ = val.value(df, **kw)

                    if switching_on_ is not None:
                        cond_ = switching_on_ == cond_

                    result.mask(cond_, val_, inplace=True)

                return result

            result_dtype, result_ftype = self._translate_case_common(
                expr, switching_on, cases, default, **kwargs
            )
            return TypedValue(value, result_dtype, result_ftype)

        def arranged_window(
            self,
            value: Callable,
            operator: ops.Operator,
            context_kwargs: dict,
            **translate_kwargs,
        ):
            arrange = context_kwargs.get("arrange")
            if arrange is None:
                warnings.warn("Argument 'arrange' is required with SQL backend.")
                return value
            ordering = translate_ordering(self.backend, arrange)

            def arranged_value(df, **kwargs):
                original_index = df.index
                sorted_df = self.backend.sort_df(df, ordering, translate_kwargs)

                # Original grouper can't be used for sorted dataframe
                sorted_grouper = self.backend.df_grouper(sorted_df)
                kwargs["grouper"] = sorted_grouper

                v = value(sorted_df, **kwargs)
                v_with_original_index = v.loc[original_index]
                return v_with_original_index

            return arranged_value

    class AlignedExpressionEvaluator(
        EagerTableImpl.AlignedExpressionEvaluator[TypedValue[pd.Series]]
    ):
        def _translate_col(self, expr, **kwargs):
            df_col_name = expr.table.df_name_mapping[expr.uuid]
            series = expr.table.df[df_col_name]
            dtype = expr.table.cols[expr.uuid].dtype
            return TypedValue(series, dtype)

        def _translate_literal_col(self, expr, **kwargs):
            assert issubclass(expr.backend, PandasTableImpl)
            return expr.typed_value

        def _translate_function(
            self, expr, implementation, op_args, context_kwargs, **kwargs
        ):
            # Drop index to evaluate aligned
            # TODO: Maybe we can somehow restore the index at the end.
            arguments = [
                arg.value.reset_index(drop=True)
                if isinstance(arg.value, pd.Series)
                else arg.value
                for arg in op_args
            ]

            # Check that they have the same length
            series_lengths = {
                len(arg.index) for arg in arguments if isinstance(arg, pd.Series)
            }
            if len(series_lengths) >= 2:
                arg_lengths = [
                    len(arg.index) if isinstance(arg, pd.Series) else 1
                    for arg in arguments
                ]
                raise AlignmentError(
                    f"Arguments for function {expr.name} aren't aligned. Specifically,"
                    f" the inputs are of length {arg_lengths}. Instead they must either"
                    " all be of the same length or of length 1."
                )

            # Compute value
            operator = implementation.operator

            if operator.ftype == OPType.WINDOW and "arrange" in context_kwargs:
                raise NotImplementedError(
                    "The 'arrange' argument for window functions currently is not"
                    " supported for aligned expressions."
                )

            value = implementation(
                *arguments,
                _tbl=None,
                _df=None,
            )
            override_ftype = (
                OPType.WINDOW if operator.ftype == OPType.AGGREGATE else None
            )
            ftype = PandasTableImpl._get_op_ftype(op_args, operator, override_ftype)
            return TypedValue(value, implementation.rtype, ftype)

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
                f"Invalid ON clause element: {expr}. Only a conjunction of equalities"
                " is supported by pandas (ands of equals)."
            )

    #### HELPERS ####

    def sort_df(self, df, ordering: list[OrderingDescriptor], translate_kwargs=None):
        ascending = [o.asc for o in ordering]
        nulls_first = [o.nulls_first for o in ordering]

        if all(nulls_first):
            na_position = "first"
        elif not any(nulls_first):
            na_position = "last"
        else:
            raise NotImplementedError(
                "Pandas sort can't handle different null positions (first / last)"
                " inside a single sort. This can be resolved by splitting the ordering"
                " into multiple arranges."
            )

        # Add temporary columns
        ordering_cols_names = []
        temporary_ordering_cols = {}
        for o in ordering:
            if isinstance(o.order, Column):
                ordering_cols_names.append(self.df_name_mapping[o.order.uuid])
                continue

            # Order is an expression: Create a temporary column
            tmp_name = "tmp_arrange_" + uuid_to_str(uuid.uuid1())
            tmp_col = self.compiler.translate(
                o.order, **(translate_kwargs or {})
            ).value(df)

            ordering_cols_names.append(tmp_name)
            temporary_ordering_cols[tmp_name] = tmp_col

        df = df.copy(deep=False)
        for col_name, col in temporary_ordering_cols.items():
            df[col_name] = col

        df = df.sort_values(
            by=ordering_cols_names,
            ascending=ascending,
            kind="mergesort",
            na_position=na_position,
        )
        df.drop(columns=list(temporary_ordering_cols), inplace=True)

        return df

    def grouped_df(self, df: pd.DataFrame = None):
        if df is None:
            df = self.df

        if self.grouped_by:
            grouping_cols_names = [
                self.df_name_mapping[col.uuid] for col in self.grouped_by
            ]
            return df.groupby(by=list(grouping_cols_names), dropna=False)
        return df

    def df_grouper(self, df: pd.DataFrame = None):
        if self.grouped_by:
            return self.grouped_df(df).grouper
        return None


def fast_pd_convert_dtypes(obj: pd._typing.NDFrameT, **kwargs) -> pd._typing.NDFrameT:
    """
    Faster alternative to the pandas `DataFrame.convert_dtypes` method.
    Unlike the builtin version, this function tries to minimize unnecessary
    copy operations.

    As long as all columns are already of the nullable pandas types, this
    function is a noop.
    """

    if obj.ndim == 1:
        if isinstance(obj.dtype, pd.core.dtypes.dtypes.ExtensionDtype):
            return obj

        inferred_dtype = pandas.core.dtypes.cast.convert_dtypes(obj._values, **kwargs)
        result = obj.astype(inferred_dtype)
        return result

    if not all(
        isinstance(dtype, pd.core.dtypes.dtypes.ExtensionDtype) for dtype in obj.dtypes
    ):
        results = [
            fast_pd_convert_dtypes(col, **kwargs) for col_name, col in obj.items()
        ]

        if len(results) > 0:
            result = pd.concat(results, axis=1, copy=False, keys=obj.columns)
            result = obj._constructor(result)
            result = result.__finalize__(obj, method="convert_dtypes")
            return result

    return obj


#### BACKEND SPECIFIC OPERATORS ################################################


def _pandas_nullable_comparison(lhs, rhs, op):
    """
    Implement comparison operation between series with non-nullable dtypes.
    """
    x = op(lhs, rhs).astype(pd.BooleanDtype(), copy=False)
    if isinstance(lhs, pd.Series):
        x.mask(lhs.isna(), pd.NA, inplace=True)
    if isinstance(rhs, pd.Series):
        x.mask(rhs.isna(), pd.NA, inplace=True)
    return x


with PandasTableImpl.op(ops.Equal()) as op:

    @op("T, const none -> bool")
    def _eq(lhs, rhs):
        assert rhs is None
        return lhs.isna()

    @op("datetime, datetime -> bool")
    def _eq(lhs, rhs):
        return _pandas_nullable_comparison(lhs, rhs, py_operator.eq)


with PandasTableImpl.op(ops.NotEqual()) as op:

    @op("T, const none -> bool")
    def _ne(lhs, rhs):
        assert rhs is None
        return lhs.notna()

    @op("datetime, datetime -> bool")
    def _ne(lhs, rhs):
        return _pandas_nullable_comparison(lhs, rhs, py_operator.ne)


with PandasTableImpl.op(ops.Less()) as op:

    @op("datetime, datetime -> bool")
    def _lt(lhs, rhs):
        return _pandas_nullable_comparison(lhs, rhs, py_operator.lt)


with PandasTableImpl.op(ops.LessEqual()) as op:

    @op("datetime, datetime -> bool")
    def _le(lhs, rhs):
        return _pandas_nullable_comparison(lhs, rhs, py_operator.le)


with PandasTableImpl.op(ops.Greater()) as op:

    @op("datetime, datetime -> bool")
    def _gt(lhs, rhs):
        return _pandas_nullable_comparison(lhs, rhs, py_operator.gt)


with PandasTableImpl.op(ops.GreaterEqual()) as op:

    @op("datetime, datetime -> bool")
    def _ge(lhs, rhs):
        return _pandas_nullable_comparison(lhs, rhs, py_operator.ge)


with PandasTableImpl.op(ops.Pow()) as op:

    @op("int, int -> float")
    def _pow(lhs, rhs):
        if isinstance(lhs, int):
            return lhs ** rhs.astype(pd.Float64Dtype())
        return lhs.astype(pd.Float64Dtype()) ** rhs


with PandasTableImpl.op(ops.RPow()) as op:

    @op("int, int -> float")
    def _rpow(rhs, lhs):
        return _pow(lhs, rhs)


with PandasTableImpl.op(ops.Round()) as op:

    @op.auto
    def _round(x, decimals=0):
        return x.round(decimals=decimals)


with PandasTableImpl.op(ops.IsIn()) as op:

    @op.auto
    def _isin(x, *values):
        y = x.isin(values)
        y.mask(x.isna(), pd.NA, inplace=True)
        return y

    @op("datetime, const datetime... -> bool")
    def _isin(x, *values):
        y = x.isin(values).astype(pd.BooleanDtype(), copy=False)
        y.mask(x.isna(), pd.NA, inplace=True)
        return y


#### String Functions ####


with PandasTableImpl.op(ops.Strip()) as op:

    @op.auto
    def _strip(x):
        return x.str.strip()


with PandasTableImpl.op(ops.StringLength()) as op:

    @op.auto
    def _str_length(x):
        return x.str.len()


with PandasTableImpl.op(ops.Upper()) as op:

    @op.auto
    def _upper(x):
        return x.str.upper()


with PandasTableImpl.op(ops.Lower()) as op:

    @op.auto
    def _upper(x):
        return x.str.lower()


with PandasTableImpl.op(ops.Replace()) as op:

    @op.auto
    def _replace(x, y, z):
        return x.str.replace(y, z)


with PandasTableImpl.op(ops.StartsWith()) as op:

    @op.auto
    def _startswith(x, y):
        return x.str.startswith(y)


with PandasTableImpl.op(ops.EndsWith()) as op:

    @op.auto
    def _endswith(x, y):
        return x.str.endswith(y)


with PandasTableImpl.op(ops.Contains()) as op:

    @op.auto
    def _contains(x, y):
        return x.str.contains(y)


#### Datetime Functions ####


with PandasTableImpl.op(ops.Year()) as op:

    @op.auto
    def _year(x):
        return x.dt.year


with PandasTableImpl.op(ops.Month()) as op:

    @op.auto
    def _month(x):
        return x.dt.month


with PandasTableImpl.op(ops.Day()) as op:

    @op.auto
    def _day(x):
        return x.dt.day


with PandasTableImpl.op(ops.Hour()) as op:

    @op.auto
    def _hour(x):
        return x.dt.hour


with PandasTableImpl.op(ops.Minute()) as op:

    @op.auto
    def _minute(x):
        return x.dt.minute


with PandasTableImpl.op(ops.Second()) as op:

    @op.auto
    def _second(x):
        return x.dt.second


with PandasTableImpl.op(ops.Millisecond()) as op:

    @op.auto
    def _millisecond(x):
        return x.dt.microsecond // 1000


with PandasTableImpl.op(ops.DayOfWeek()) as op:

    @op.auto
    def _day_of_week(x):
        return x.dt.day_of_week + 1


with PandasTableImpl.op(ops.DayOfYear()) as op:

    @op.auto
    def _day_of_year(x):
        return x.dt.day_of_year


#### Summarising Functions ####


with PandasTableImpl.op(ops.Mean()) as op:

    @op.auto
    def _mean(x):
        return x.mean()

    @op.auto(variant="transform")
    def _mean(x):
        return x.transform("mean")

    @op.auto(variant="aggregate")
    def _mean(x):
        return x.aggregate("mean")


with PandasTableImpl.op(ops.Min()) as op:

    @op.auto
    def _min(x):
        return x.min()

    @op.auto(variant="transform")
    def _min(x):
        return x.transform("min")

    @op.auto(variant="aggregate")
    def _min(x):
        return x.aggregate("min")


with PandasTableImpl.op(ops.Max()) as op:

    @op.auto
    def _max(x):
        return x.max()

    @op.auto(variant="transform")
    def _max(x):
        return x.transform("max")

    @op.auto(variant="aggregate")
    def _max(x):
        return x.aggregate("max")


with PandasTableImpl.op(ops.Sum()) as op:

    @op.auto
    def _sum(x):
        return x.sum()

    @op.auto(variant="transform")
    def _sum(x):
        return x.transform("sum")

    @op.auto(variant="aggregate")
    def _sum(x):
        return x.aggregate("sum")


with PandasTableImpl.op(ops.Any()) as op:

    @op.auto
    def _any(x):
        return x.any()

    @op.auto(variant="transform")
    def _any(x):
        return x.transform("any")

    @op.auto(variant="aggregate")
    def _any(x):
        return x.aggregate("any")


with PandasTableImpl.op(ops.All()) as op:

    @op.auto
    def _all(x):
        return x.all()

    @op.auto(variant="transform")
    def _all(x):
        return x.transform("all")

    @op.auto(variant="aggregate")
    def _all(x):
        return x.aggregate("all")


with PandasTableImpl.op(ops.StringJoin()) as op:

    @op.auto
    def _join(x, sep):
        # Must remove nulls
        return sep.join(x[x.notnull()])


with PandasTableImpl.op(ops.Count()) as op:

    @op.auto(variant="aggregate")
    @op.auto
    def _count(x: pd.Series):
        # Count non null values
        return x.count()

    @op.auto(variant="transform")
    def _count(x):
        return x.transform("count")


#### Window Functions ####

with PandasTableImpl.op(ops.Shift()) as op:

    @op.auto(variant="transform")
    @op.auto
    def _shift(x: pd.Series, by, empty_value=None):
        return x.shift(by, fill_value=empty_value)


with PandasTableImpl.op(ops.RowNumber()) as op:

    @op.auto
    def _row_number(idx: pd.Series):
        n = len(idx.index)
        return pd.Series(np.arange(1, n + 1), index=idx.index)

    @op.auto(variant="transform")
    def _row_number(idx: pd.Series):
        return idx.cumcount() + 1


with PandasTableImpl.op(ops.Rank()) as op:

    @op.auto
    def _rank(x: pd.Series, *, _ordering: list[OrderingDescriptor]):
        assert len(_ordering) == 1
        ordering = _ordering[0]
        na_option = "top" if ordering.nulls_first else "bottom"

        return x.rank(method="min", ascending=ordering.asc, na_option=na_option)

    # transform variant currently disabled due to bug in pandas:
    # https://github.com/pandas-dev/pandas/issues/54206

    # @op.auto(variant="transform")
    # def _rank(x: pd.Series, *, _ordering: list[OrderingDescriptor]):
    #     assert len(_ordering) == 1
    #     ordering = _ordering[0]
    #     na_option = "top" if ordering.nulls_first else "bottom"
    #     return x.transform(
    #         "rank", method="min", ascending=ordering.asc, na_option=na_option
    #     )


with PandasTableImpl.op(ops.DenseRank()) as op:

    @op.auto
    def _dense_rank(x: pd.Series, *, _ordering: list[OrderingDescriptor]):
        assert len(_ordering) == 1
        ordering = _ordering[0]
        na_option = "top" if ordering.nulls_first else "bottom"

        return x.rank(method="dense", ascending=ordering.asc, na_option=na_option)

    # transform variant currently disabled due to bug in pandas:
    # https://github.com/pandas-dev/pandas/issues/54206

    # @op.auto(variant="transform")
    # def _dense_rank(x: pd.Series, *, _ordering: list[OrderingDescriptor]):
    #     assert len(_ordering) == 1
    #     ordering = _ordering[0]
    #     na_option = "top" if ordering.nulls_first else "bottom"
    #     return x.transform(
    #         "rank", method="dense", ascending=ordering.asc, na_option=na_option
    #     )
