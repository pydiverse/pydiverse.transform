import functools
import itertools
import operator
from typing import Callable

import numpy as np
import pandas as pd

from pdtransform.core.column import Column
from pdtransform.core.expressions import FunctionCall, SymbolicExpression
from pdtransform.core.expressions.translator import Translator, TypedValue
from .eager_table import EagerTableImpl, uuid_to_str


class PandasTableImpl(EagerTableImpl):
    """Pandas backend

    Attributes:
        df: The current, eager dataframe. It's columns get renamed to prevent
            name collisions and it also contains all columns that have been
            computed at some point. This allows for a more lazy style API.
    """

    def __init__(self, name: str, df: pd.DataFrame):
        self.df = df
        self.join_translator = self.JoinTranslator(self)

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

    def _bind_values_to_compiled_expr(self, compiled):
        return lambda x: compiled(self.df)

    #### Verb Operations ####

    def alias(self, name):
        # Creating a new table object also acts like a garbage collecting mechanism.
        return self.__class__(name, self.collect())

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

    def join(self, right: 'PandasTableImpl', on: SymbolicExpression, how: str):
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

        # Do joining -> Look at siuba implementation
        if how is None:
            raise Exception("Must specify how argument")

        left_on, right_on = zip(*on_cols)
        left_on  = [self.df_name_mapping[col.uuid] for col in left_on]
        right_on = [right.df_name_mapping[col.uuid] for col in right_on]

        self.df_name_mapping.update(right.df_name_mapping)
        self.df = self.df.merge(right.df, how = how, left_on = left_on, right_on = right_on)

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

    class ExpressionCompiler(Translator['PandasTableImpl', TypedValue[Callable[[pd.DataFrame], pd.Series]]]):

        def _translate(self, expr, verb=None, **kwargs):
            if isinstance(expr, Column):
                df_col_name = self.backend.df_name_mapping[expr.uuid]
                def df_col(df):
                    return df[df_col_name]
                return TypedValue(df_col, expr.dtype)

            if isinstance(expr, FunctionCall):
                arguments = [arg.value for arg in expr.args]
                signature = tuple(arg.dtype for arg in expr.args)
                implementation = self.backend.operator_registry.get_implementation(expr.operator, signature)

                def value(df):
                    return implementation(*(arg(df) for arg in arguments))

                override_impl_ftype = 'w' if implementation.ftype == 'a' and verb == 'mutate' else None
                ftype = self.backend._get_func_ftype(expr.args, implementation, override_impl_ftype)
                return TypedValue(value, implementation.rtype, ftype)

            if isinstance(expr, TypedValue):
                # For iPython formatting
                return expr

            # Literals
            def literal_func(_):
                return expr

            if isinstance(expr, int):
                return TypedValue(literal_func, 'int')
            if isinstance(expr, float):
                return TypedValue(literal_func, 'float')
            if isinstance(expr, str):
                return TypedValue(literal_func, 'str')
            if isinstance(expr, bool):
                return TypedValue(literal_func, 'bool')

            raise NotImplementedError(expr, type(expr))

    class JoinTranslator(Translator['PandasTableImpl', tuple]):
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
            return self.df.groupby(by = list(grouping_cols_names))
        return self.df


#### BACKEND SPECIFIC OPERATORS ################################################

...

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
