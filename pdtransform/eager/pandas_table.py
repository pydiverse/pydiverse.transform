import itertools
import functools
import operator

import pandas as pd
import numpy as np

from .eager_table import EagerTableImpl, uuid_to_str
from pdtransform.core.expressions.translator import Translator, TypedValue
from pdtransform.core.column import Column
from pdtransform.core.expressions import FunctionCall, SymbolicExpression
from ..core.expressions.lambda_column import LambdaColumn


class PandasTableImpl(EagerTableImpl):

    def __init__(self, name: str, df: pd.DataFrame):
        self.df = df
        self.translator = PandasExpressionTranslator(self)
        self.join_translator = PandasJoinTranslator(self)

        cols = self.df.dtypes.items()
        super().__init__(
            name = name,
            columns = {
                name: Column(name = name, table = self, dtype = self._convert_dtype(dtype))
                for name, dtype in cols
            }
        )

        # Rename columns
        self.df_name_mapping = {
            col._uuid: f'{self.name}_{col._name}_' + uuid_to_str(col._uuid)
            for col in self.columns.values()
        }  # type: dict[uuid.UUID: str]

        self.df = self.df.rename(columns = {
            name: self.df_name_mapping[col._uuid]
            for name, col in self.columns.items()
        })

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

    def copy(self):
        c = super().copy()
        # Must create a new translator, so that it can access the current df.
        c.translator = PandasExpressionTranslator(c)
        return c

    #### Verb Operations ####

    def collect(self):
        # SELECT -> Apply mask
        selected_cols_name_map = { self.df_name_mapping[self.named_cols.fwd[name]]: name for name in self.selects }
        masked_df = self.df[[*selected_cols_name_map.keys()]]
        return masked_df.rename(columns = selected_cols_name_map)

    def mutate(self, **kwargs):
        uuid_kwargs = { self.named_cols.fwd[k]: (k, v) for k, v in kwargs.items() }
        self.df_name_mapping.update({uuid: f'{self.name}_mutate_{name}_' + uuid_to_str(uuid) for uuid, (name, _) in uuid_kwargs.items() })
        typed_cols = { uuid: self.translator.translate(expr) for uuid, (_, expr) in uuid_kwargs.items() }

        # Update Dataframe
        cols = {self.df_name_mapping[uuid]: v.value for uuid, v in typed_cols.items()}
        self.df = self.df.assign(**cols)

        # And update dtype metadata
        cols_dtypes = { uuid: v.dtype for uuid, v in typed_cols.items() }
        self.col_dtype.update(cols_dtypes)

    def join(self, right: 'PandasTableImpl', on: SymbolicExpression, how: str):
        """
        :param on: Symbolic expression consisting of anded column equality checks.
        """

        # Parse ON condition
        on_cols = []
        for col1, col2 in self.join_translator.translate(on):
            if col1._uuid in self.col_expr and col2._uuid in right.col_expr:
                on_cols.append((col1, col2))
            elif col2._uuid in self.col_expr and col1._uuid in right.col_expr:
                on_cols.append((col2, col1))
            else:
                raise Exception

        # Do joining -> Look at siuba implementation
        if how is None:
            raise Exception("Must specify how argument")

        left_on, right_on = zip(*on_cols)
        left_on  = [self.df_name_mapping[col._uuid] for col in left_on]
        right_on = [right.df_name_mapping[col._uuid] for col in right_on]

        self.df_name_mapping.update(right.df_name_mapping)
        self.df = self.df.merge(right.df, how = how, left_on = left_on, right_on = right_on)

    def filter(self, *args: SymbolicExpression):
        if not args:
            return

        condition, cond_type = self.translator.translate(functools.reduce(operator.and_, args))
        assert(cond_type == 'bool')

        self.df = self.df.loc[condition]


class PandasExpressionTranslator(Translator):

    def _translate(self, expr):
        if isinstance(expr, Column):
            df_col_name = self.backend.df_name_mapping[expr._uuid]
            df_col = self.backend.df[df_col_name]
            return TypedValue(df_col, expr._dtype)

        if isinstance(expr, FunctionCall):
            arguments = [arg.value for arg in expr._args]
            signature = tuple(arg.dtype for arg in expr._args)
            implementation = self.backend.operator_registry.get_implementation(expr._operator, signature)
            return TypedValue(implementation(*arguments), implementation.rtype)

        # Literals
        if isinstance(expr, int):
            return TypedValue(expr, 'int')
        if isinstance(expr, str):
            return TypedValue(expr, 'str')

        raise NotImplementedError(expr, type(expr))


class PandasJoinTranslator(Translator):
    """
    This translator takes a conjunction (AND) of equality checks and returns
    a tuple of tuple where the inner tuple contains the left and right column
    of the equality checks.
    """
    def _translate(self, expr):
        if isinstance(expr, Column):
            return expr
        if isinstance(expr, FunctionCall):
            if expr._operator == '__eq__':
                c1 = expr._args[0]
                c2 = expr._args[1]
                assert(isinstance(c1, Column) and isinstance(c2, Column))
                return ((c1, c2),)
            if expr._operator == '__and__':
                return tuple(itertools.chain(*expr._args))
        raise Exception(f'Invalid ON clause element: {expr}. Only a conjunction of equalities is supported by pandas (ands of equals).')



@PandasTableImpl.register_op('str.strip', 'object -> object')
def _str_strip(x: pd.Series):
    return x.str.strip()
