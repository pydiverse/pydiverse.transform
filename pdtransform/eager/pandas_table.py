import functools
import itertools
import operator
from typing import Callable

import numpy as np
import pandas as pd

from pdtransform.core.column import Column, LiteralColumn
from pdtransform.core.expressions import FunctionCall, SymbolicExpression
from pdtransform.core.expressions.translator import Translator, TypedValue
from .eager_table import EagerTableImpl, uuid_to_str


class PandasTableImpl(EagerTableImpl):
    """Pandas backend

    Attributes:
        df: The current, eager dataframe. It's columns get renamed to prevent
            name collisions, and it also contains all columns that have been
            computed at some point. This allows for a more lazy style API.
    """

    def __init__(self, name: str, df: pd.DataFrame):
        self.df = df
        self.join_translator = self.JoinTranslator()

        columns = {
            name: Column(name = name, table = self, dtype = self._convert_dtype(dtype))
            for name, dtype in self.df.dtypes.items()
        }

        # Rename columns
        self.df_name_mapping = {
            col.uuid: f'{name}_{col.name}_' + uuid_to_str(col.uuid)
            for col in columns.values()
        }  # type: dict[uuid.UUID: str]

        self.df = self.df.rename(columns = {
            name: self.df_name_mapping[col.uuid]
            for name, col in columns.items()
        })

        super().__init__(name = name, columns = columns)

    def _convert_dtype(self, dtype: np.dtype) -> str:
        # b  boolean
        # i  signed integer
        # u  unsigned integer
        # f  floating-point
        # c  complex floating-point
        # m  timedelta
        # M  datetime
        # O  object
        # S  (byte-)string
        # U  Unicode
        # V  void

        kind_map = {
            'b': 'bool',
            'i': 'int',
            'f': 'float',
            'O': 'object',  # TODO: Determine actual type
        }

        return kind_map.get(dtype.kind, str(dtype))

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
        selected_cols_name_map = { self.df_name_mapping[uuid]: name for name, uuid in self.selected_cols() }
        masked_df = self.df[[*selected_cols_name_map.keys()]]

        # rename columns from internal naming scheme to external names
        return masked_df.rename(columns = selected_cols_name_map)

    def mutate(self, **kwargs):
        uuid_kwargs = { self.named_cols.fwd[k]: (k, v) for k, v in kwargs.items() }
        self.df_name_mapping.update({
            uuid: f'{self.name}_mutate_{name}_' + uuid_to_str(uuid)
            for uuid, (name, _) in uuid_kwargs.items()
        })

        # Update Dataframe
        if self.grouped_by:
            # Window Functions
            gdf = self.grouped_df()
            cols_transforms = {
                self.df_name_mapping[uuid]: self.cols[uuid].compiled
                for uuid in uuid_kwargs.keys()
            }
            self.df = gdf.apply(lambda x: x.assign(**{k: v(x) for k, v in cols_transforms.items()}))
        else:
            # Normal Functions
            cols = {
                self.df_name_mapping[uuid]: self.cols[uuid].compiled(self.df)
                for uuid in uuid_kwargs.keys()
            }
            self.df = self.df.assign(**cols)

    def join(self, right: 'PandasTableImpl', on: SymbolicExpression, how: str, *, validate=None):
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
            validate = {
                '1:?': '1:1'
            }[validate]

        # Perform Join
        left_on, right_on = zip(*on_cols)
        left_on  = [self.df_name_mapping[col.uuid] for col in left_on]
        right_on = [right.df_name_mapping[col.uuid] for col in right_on]

        self.df_name_mapping.update(right.df_name_mapping)
        self.df = self.df.merge(
            right.df, how = how,
            left_on = left_on, right_on = right_on,
            validate = validate)

    def filter(self, *args: SymbolicExpression):
        if not args:
            return

        compiled, dtype = self.compiler.translate(functools.reduce(operator.and_, args))
        assert(dtype == 'bool')

        condition = compiled(self.df)
        self.df = self.df.loc[condition]

    def arrange(self, ordering: list[tuple[SymbolicExpression, bool]]):
        cols, ascending = zip(*ordering)
        cols = [self.df_name_mapping[col.uuid] for col in cols]
        self.df = self.df.sort_values(by = cols, ascending = ascending, kind = 'mergesort')

    def summarise(self, **kwargs):
        translated_values = {}

        grouped_df = self.grouped_df()
        for name, expr in kwargs.items():
            uuid = self.named_cols.fwd[name]
            internal_name = f'{self.name}_summarise_{name}_{uuid_to_str(uuid)}'
            self.df_name_mapping[uuid] = internal_name

            compiled = self.cols[uuid].compiled
            if self.grouped_by:
                translated_values[internal_name] = grouped_df.apply(compiled)
            else:
                translated_values[internal_name] = compiled(grouped_df)

        # Grouped Dataframe requires different operations compared to ungrouped df.
        if self.grouped_by:
            columns = { k: v.rename(k) for k, v in translated_values.items() }
            self.df = pd.concat(columns, axis = 'columns').reset_index()
        else:
            self.df = pd.DataFrame(translated_values, index = [0])

    #### EXPRESSIONS ####

    class ExpressionCompiler(EagerTableImpl.ExpressionCompiler['PandasTableImpl', TypedValue[Callable[[pd.DataFrame], pd.Series]]]):

        def _translate_col(self, expr, **kwargs):
            df_col_name = self.backend.df_name_mapping[expr.uuid]
            def df_col(df):
                return df[df_col_name]
            return TypedValue(df_col, expr.dtype)

        def _translate_literal_col(self, expr, **kwargs):
            if not self.backend.is_aligned_with(expr):
                raise ValueError(f"Literal column isn't aligned with this table. "
                                 f"Literal Column: {expr}")

            def value(df):
                literal_value = expr.typed_value.value
                if isinstance(literal_value, pd.Series):
                    literal_len = len(literal_value.index)
                    if len(df.index) != literal_len:
                        raise ValueError(f"Literal column isn't aligned with this table. "
                                         f"Make sure that they have the same length ({len(df.index)} vs {literal_len}). "
                                         f"This might be the case if you are using literal columns inside a window function.\n"
                                         f"Literal Column: {expr}")

                    series = literal_value.copy()
                    series.index = df.index
                    return series
                return literal_value

            return TypedValue(value, expr.typed_value.dtype, expr.typed_value.ftype)

        def _translate_function(self, expr, arguments, implementation, verb=None, **kwargs):
            def value(df):
                return implementation(*(arg(df) for arg in arguments))

            override_impl_ftype = 'w' if implementation.ftype == 'a' and verb == 'mutate' else None
            ftype = self.backend._get_func_ftype(expr.args, implementation, override_impl_ftype)
            return TypedValue(value, implementation.rtype, ftype)

    class AlignedExpressionEvaluator(EagerTableImpl.AlignedExpressionEvaluator[TypedValue[pd.Series]]):

        def _translate_col(self, expr, **kwargs):
            df_col_name = expr.table.df_name_mapping[expr.uuid]
            series = expr.table.df[df_col_name]
            dtype = expr.table.cols[expr.uuid].dtype
            return TypedValue(series, dtype)

        def _translate_literal_col(self, expr, **kwargs):
            assert issubclass(expr.backend, PandasTableImpl)
            return expr.typed_value

        def _translate_function(self, expr, arguments, implementation, **kwargs):
            # Drop index to evaluate aligned
            # TODO: Maybe we can somehow restore the index at the end.
            arguments = [arg.reset_index(drop = True) if isinstance(arg, pd.Series) else arg for arg in arguments]

            # Check that they have the same length
            series_lengths = { len(arg.index) for arg in arguments if isinstance(arg, pd.Series) }
            if len(series_lengths) >= 2:
                arg_lengths = [len(arg.index) if isinstance(arg, pd.Series) else 1 for arg in arguments]
                raise ValueError(f"Arguments for function {expr.operator} aren't aligned. Specifically, the inputs are of lenght {arg_lengths}. Instead they must either all be of the same length or of length 1.")

            value = implementation(*arguments)
            override_impl_ftype = 'w' if implementation.ftype == 'a' else None
            ftype = PandasTableImpl._get_func_ftype(expr.args, implementation, override_impl_ftype)
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
                if expr.operator == '__eq__':
                    c1 = expr.args[0]
                    c2 = expr.args[1]
                    assert(isinstance(c1, Column) and isinstance(c2, Column))
                    return ((c1, c2),)
                if expr.operator == '__and__':
                    return tuple(itertools.chain(*expr.args))
            raise Exception(f'Invalid ON clause element: {expr}. Only a conjunction of equalities is supported by pandas (ands of equals).')

    #### GROUPING HELPER ####

    def grouped_df(self):
        if self.grouped_by:
            grouping_cols_names = [self.df_name_mapping[col.uuid] for col in self.grouped_by]
            return self.df.groupby(by = list(grouping_cols_names), dropna=False)
        return self.df


#### BACKEND SPECIFIC OPERATORS ################################################

@PandasTableImpl.op('__round__', 'int -> int')
@PandasTableImpl.op('__round__', 'int, int -> int')
def _round(x, decimals=0):
    # Int is already rounded
    return x

@PandasTableImpl.op('__round__', 'float -> float')
@PandasTableImpl.op('__round__', 'float, int -> float')
def _round(x, decimals=0):
    return x.round(decimals=decimals)

#### Summarising Functions ####

@PandasTableImpl.op('mean', 'int |> float')
@PandasTableImpl.op('mean', 'float |> float')
def _mean(x):
    return x.mean()

@PandasTableImpl.op('min', 'int |> float')
@PandasTableImpl.op('min', 'float |> float')
@PandasTableImpl.op('min', 'str |> str')
def _min(x):
    return x.min()

@PandasTableImpl.op('max', 'int |> float')
@PandasTableImpl.op('max', 'float |> float')
@PandasTableImpl.op('max', 'str |> str')
def _max(x):
    return x.max()

@PandasTableImpl.op('sum', 'int |> float')
@PandasTableImpl.op('sum', 'float |> float')
def _sum(x):
    return x.sum()

@PandasTableImpl.op('count', 'T |> int')
def _count(x):
    return len(x)