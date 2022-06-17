from collections import namedtuple

from pdtransform.core.expressions import Translator
from pdtransform.core.expressions.lambda_column import LambdaColumn
from pdtransform.core.table_impl import AbstractTableImpl


JoinDescriptor = namedtuple('JoinDescriptor', ['right', 'on', 'how'])
OrderByDescriptor = namedtuple('OrderByDescriptor', ['order', 'asc', 'nulls_first'])


class LazyTableImpl(AbstractTableImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_translator = LazyLambdaTranslator(self)

    def copy(self):
        c = super().copy()
        c.lambda_translator = LazyLambdaTranslator(c)
        return c

    def resolve_lambda_cols(self, expr):
        return self.lambda_translator.translate(expr)

    def query_string(self) -> str:
        raise NotImplementedError


class LazyLambdaTranslator(Translator):
    def _translate(self, expr):
        # Replace lambda with corresponding symbolic expression
        if isinstance(expr, LambdaColumn):
            uuid = self.backend.named_cols.fwd[expr._name]
            lambda_expr = self.backend.col_expr[uuid]
            return lambda_expr
        return expr