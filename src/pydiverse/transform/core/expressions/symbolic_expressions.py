from __future__ import annotations

from html import escape
from typing import Any, Generic, TypeVar

from pydiverse.transform.core import alignment
from pydiverse.transform.core.util import traverse

from ..ops import registry
from . import expressions, util

T = TypeVar("T")


class SymbolicExpression(Generic[T]):
    """
    Base class to represent a symbolic expression. It can be manipulated using
    standard python operators (for example you can add them) or by calling
    attributes of it.

    To get the non-symbolic version of this expression you use the
    underscore `_` attribute.
    """

    __slots__ = ("_",)

    def __init__(self, underlying: T):
        self._ = underlying

    def __getattr__(self, item) -> SymbolAttribute:
        if item.startswith("_") and item.endswith("_") and len(item) >= 3:
            # Attribute names can't begin and end with an underscore because
            # IPython calls hasattr() to select the correct pretty printing
            # function. Instead of hard coding a specific list, just throw
            # an exception for all attributes that match the general pattern.
            raise AttributeError(
                f"Invalid attribute {item}. Attributes can't begin and end with an"
                " underscore."
            )
        return SymbolAttribute(item, self)

    def __getitem__(self, item):
        return SymbolicExpression(expressions.FunctionCall("__getitem__", self, item))

    def __dir__(self):
        # TODO: Instead of displaying all available operators, translate the
        #       expression and according to the dtype and backend only display
        #       the operators that actually are available.
        return sorted(registry.OperatorRegistry.ALL_REGISTERED_OPS)

    # __contains__, __iter__ and __bool__ are all invalid on s-expressions
    __contains__ = None
    __iter__ = None

    def __bool__(self):
        raise TypeError(
            "Symbolic expressions can't be converted to True/False, "
            "or used with these keywords: not, and, or."
        )

    def __str__(self):
        try:
            result = alignment.eval_aligned(self._, check_alignment=False)._

            dtype = result.typed_value.dtype
            value = result.typed_value.value
            return (
                f"Symbolic Expression: {repr(self._)}\ndtype: {dtype}\n\n{str(value)}"
            )
        except Exception as e:
            return (
                f"Symbolic Expression: {repr(self._)}\n"
                "Failed to get evaluate due to an exception:\n"
                f"{type(e).__name__}: {str(e)}"
            )

    def __repr__(self):
        return f"<Sym: {self._}>"

    def _repr_html_(self):
        html = f"<pre>Symbolic Expression:\n{escape(repr(self._))}</pre>"

        try:
            result = alignment.eval_aligned(self._, check_alignment=False)._
            backend = util.determine_expr_backend(self._)

            value_repr = backend._html_repr_expr(result.typed_value.value)
            html += f"dtype: <code>{escape(result.typed_value.dtype)}</code></br></br>"
            html += f"<pre>{escape(value_repr)}</pre>"
        except Exception as e:
            html += (
                "</br><pre>Failed to get evaluate due to an exception:\n"
                f"{escape(e.__class__.__name__)}: {escape(str(e))}</pre>"
            )

        return html

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")


class SymbolAttribute:
    def __init__(self, name: str, on: SymbolicExpression):
        self.__name = name
        self.__on = on

    def __getattr__(self, item) -> SymbolAttribute:
        return SymbolAttribute(self.__name + "." + item, self.__on)

    def __call__(self, *args, **kwargs) -> SymbolicExpression:
        return SymbolicExpression(
            expressions.FunctionCall(self.__name, self.__on, *args, **kwargs)
        )

    def __hash__(self):
        raise Exception(
            "Nope... You probably didn't want to do this. Did you misspell the"
            f" attribute name '{self.__name}' of '{self.__on}'? Maybe you forgot a"
            " leading underscore."
        )


def unwrap_symbolic_expressions(arg: Any = None):
    """
    Replaces all symbolic expressions in the input with their underlying value.
    """
    return traverse(arg, lambda x: x._ if isinstance(x, SymbolicExpression) else x)


# Add all supported dunder methods to `SymbolicExpression`.
# This has to be done, because Python doesn't call __getattr__ for
# dunder methods.
def create_operator(op):
    def impl(*args, **kwargs):
        return SymbolicExpression(expressions.FunctionCall(op, *args, **kwargs))

    return impl


for dunder in registry.OperatorRegistry.SUPPORTED_DUNDER:
    setattr(SymbolicExpression, dunder, create_operator(dunder))
del create_operator
