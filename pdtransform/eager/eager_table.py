from pdtransform.core import Column
from pdtransform.core.expressions import Translator
from pdtransform.core.expressions.lambda_column import LambdaColumn
from pdtransform.core.table_impl import AbstractTableImpl


def uuid_to_str(_uuid):
    # mod with 2^31-1  (prime number)
    return format(_uuid.int % 0x7FFFFFFF, 'X')


class EagerTableImpl(AbstractTableImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_translator = EagerLambdaTranslator(self)

    def copy(self):
        c = super().copy()
        c.lambda_translator = EagerLambdaTranslator(c)
        return c

    def resolve_lambda_cols(self, expr):
        return self.lambda_translator.translate(expr)


class EagerLambdaTranslator(Translator):
    def _translate(self, expr):
        # Resolve lambda and return Column object
        if isinstance(expr, LambdaColumn):
            uuid = self.backend.named_cols.fwd[expr._name]
            dtype = self.backend.col_dtype[uuid]

            return Column(
                name = 'λ_' + expr._name,
                table = self.backend,
                dtype = dtype,
                uuid = uuid
            )
        return expr