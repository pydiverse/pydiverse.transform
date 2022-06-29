from html import escape
from typing import Any, Generic, TypeVar

from pdtransform.core import column
from . import expressions, utils, translator
from .operator_registry import OperatorRegistry

T = TypeVar('T')
class SymbolicExpression(Generic[T]):
    """
    Base class to represent a symbolic expression. It can be manipulated using
    standard python operators (for example you can add them) or by calling
    attributes of it.

    To get the non-symbolic version of this expression you use the
    underscore `_` attribute.
    """

    __slots__ = ('_', )

    def __init__(self, underlying: T):
        self._ = underlying

    def __getattr__(self, item):
        if item.startswith('_') and item.endswith('_') and len(item) >= 3:
            # Attribute names can't begin and end with an underscore because
            # IPython calls hasattr() to select the correct pretty printing
            # function. Instead of hard coding a specific list, just throw
            # an exception for all attributes that match the general pattern.
            raise AttributeError(f"Invalid attribute {item}. Attributes can't begin and end with an underscore.")
        return SymbolAttribute(item, self)

    def __getitem__(self, item):
        return SymbolicExpression(expressions.FunctionCall('__getitem__', self, item))

    def __repr__(self):
        return f'<Sym: {self._}>'

    def __dir__(self):
        # TODO: Instead of displaying all available operators, translate the
        #       expression and according to the dtype and backend only display
        #       the operators that actually are available.
        return sorted(OperatorRegistry.ALL_REGISTERED_OPS)

    def _repr_html_(self):
        html = f"<pre>Symbolic Expression:\n{escape(repr(self._))}</pre>"

        try:
            # Get Backend
            backend = None
            for item in utils.iterate_over_expr(self._):
                if isinstance(item, column.Column):
                    if backend is None:
                        backend = type(item.table)
                    elif type(item.table) != backend:
                        raise Exception(f"Can't mix different backends: {backend}, {type(item.table)}")
            if backend is None:
                raise ValueError('Failed to determine backend.')

            # Bind the correct function arguments to each column in the expression
            def base_replacer(expr):
                if isinstance(expr, column.Column):
                    tbl = expr.table
                    translated = tbl.compiler.translate(expr)
                    translated.value = tbl._bind_values_to_compiled_expr(translated.value)
                    return translated
                return expr
            base_translation = translator.bottom_up_replace(self._, base_replacer)

            # Evaluate the function calls on the shared backend
            result = backend.ExpressionCompiler(backend).translate(base_translation)
            value = result.value(None)

            value_repr = backend._html_repr_expr(value)
            html += f"dtype: <code>{escape(result.dtype)}</code></br></br>"
            html += f"<pre>{escape(value_repr)}</pre>"
        except Exception as e:
            html += f"</br><pre>Failed to get evaluate due to an exception:\n" \
                    f"{escape(e.__class__.__name__)}: {escape(str(e))}</pre>"

        return html


class SymbolAttribute:
    def __init__(self, name: str, on: SymbolicExpression):
        self.__name = name
        self.__on = on

    def __getattr__(self, item):
        return SymbolAttribute(self.__name + '.' + item, self.__on)

    def __call__(self, *args, **kwargs):
        return SymbolicExpression(expressions.FunctionCall(self.__name, self.__on, *args, **kwargs))

    def __hash__(self):
        raise Exception(f"Nope... You probably didn't want to do this. Did you misspell the attribute name '{self.__name}' of '{self.__on}'? Maybe you forgot a leading underscore.")


class ReprHTMLTranslator(translator.Translator):
    def _translate(self, expr, **kwargs):
        if isinstance(expr, column.Column):
            return expr.table.translator.translate(expr, **kwargs)
        return expr


def unwrap_symbolic_expressions(arg: Any = None):
    """
    Replaces all symbolic expressions in the input with their underlying value.
    """

    # Potential alternative approach: Iterate over object like deepcopy does

    if isinstance(arg, list):
        return [unwrap_symbolic_expressions(x) for x in arg]
    if isinstance(arg, tuple):
        if type(arg) != tuple:
            raise Exception  # This is the case with named tuples for example
        return tuple(unwrap_symbolic_expressions(x) for x in arg)
    if isinstance(arg, dict):
        return { k: unwrap_symbolic_expressions(v) for k, v in arg.items() }
    return arg._ if isinstance(arg, SymbolicExpression) else arg



# Add all supported dunder methods to `SymbolicExpression`.
# This has to be done, because Python doesn't call __getattr__ for
# dunder methods.
def create_operator(op):
    def impl(*args, **kwargs):
        return SymbolicExpression(expressions.FunctionCall(op, *args, **kwargs))
    return impl
for dunder in OperatorRegistry.SUPPORTED_DUNDER:
    setattr(SymbolicExpression, dunder, create_operator(dunder))
del create_operator
