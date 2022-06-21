import copy
import typing
import uuid

from .column import Column, LambdaColumn
from .expressions import OperatorRegistry, SymbolicExpression
from .expressions.translator import Translator, TypedValue
from .utils import bidict, ordered_set


class _TableImplMeta(type):
    """
    Metaclass that adds an appropriate operator_registry attribute to each class
    in the table implementation class hierarchy.
    """
    def __new__(cls, name, bases, attrs, **kwargs):
        c = super().__new__(cls, name, bases, attrs, **kwargs)

        # By using `super` to get the super_registry, we can check for potential
        # operation definitions in order of the MRO.
        super_reg = super(c, c).operator_registry if hasattr(super(c, c), 'operator_registry') else None
        setattr(c, 'operator_registry', OperatorRegistry(name, super_reg))
        return c


class AbstractTableImpl(metaclass=_TableImplMeta):
    """
    Base class from which all table backend implementations are derived from.
    It tracks various metadata that is relevant for all backends.
    """

    operator_registry: OperatorRegistry

    # Inner Class
    ExpressionTranslator: typing.Type[Translator['AbstractTableImpl', TypedValue]]

    def __init__(
            self,
            name: str,
            columns: dict[str, Column],
        ):

        self.name = name
        self.columns = columns
        self.translator = self.ExpressionTranslator(self)
        self.lambda_translator = self.LambdaTranslator(self)

        # selects: Ordered set of selected names.
        # named_cols: Map from name to column uuid containing all columns that have been named.
        # col_expr: Map from uuid to the `SymbolicExpression` that corresponds to this column.
        # col_dtype: Map from uuid to the datatype of the corresponding column. It is the responsibility of the backend to keep track of this information.
        self.selects = ordered_set()  # type: ordered_set[str]
        self.named_cols = bidict()    # type: bidict[str: uuid.UUID]
        self.col_expr = {}            # type: dict[uuid.UUID: SymbolicExpression]
        self.col_dtype = {}           # type: dict[uuid.UUID: str]

        self.grouped_by = []          # type: list[Column | LambdaColumn]

        # Init Values
        for name, col in columns.items():
            self.selects.add(name)
            self.named_cols.fwd[name] = col.uuid
            self.col_expr[col.uuid] = col
            self.col_dtype[col.uuid] = col.dtype

    def copy(self):
        c = copy.copy(self)
        # Copy containers
        for k, v in self.__dict__.items():
            if isinstance(v, (list, dict, set, bidict, ordered_set)):
                c.__dict__[k] = copy.copy(v)

        # Must create a new translator, so that it can access the current df.
        c.translator = self.ExpressionTranslator(c)
        c.lambda_translator = self.LambdaTranslator(c)
        return c

    def get_col(self, name: str):
        """Getter used by `Table.__getattr__`"""
        if uuid := self.named_cols.fwd.get(name, None):
            return Column(name, self, self.col_dtype.get(uuid), uuid)
        # Must return AttributeError, else `hasattr` doesn't work on Table instances.
        raise AttributeError(f"Table '{self.name}' has not column named '{name}'.")

    def selected_cols(self) -> typing.Iterable[tuple[str, uuid.UUID]]:
        for name in self.selects:
            yield (name, self.named_cols.fwd[name])

    def resolve_lambda_cols(self, expr: typing.Any):
        return self.lambda_translator.translate(expr)

    @classmethod
    def _html_repr_expr(cls, expr):
        """
        Return an appropriate string to display an expression from this backend.
        This is mainly used to IPython.
        """
        return repr(expr)

    #### Verb Callbacks ####

    def alias(self, name) -> 'AbstractTableImpl':
        ...

    def collect(self):
        ...

    def select(self, *args):
        ...

    def mutate(self, **kwargs):
        ...

    def join(self, right, on, how):
        ...

    def filter(self, *args):
        ...

    def arrange(self, ordering: list[tuple[SymbolicExpression, bool]]):
        ...

    #### Symbolic Operators ####

    @classmethod
    def register_blank_op(cls, name):
        """Register operator without providing an implementation."""
        cls.operator_registry.register_op(name, check_super = False)

    @classmethod
    def register_op(cls, name, signature):
        """Decorator: Register operator and add implementation."""
        cls.register_blank_op(name)
        return cls.op(name, signature)

    @classmethod
    def op(cls, name, signature):
        """Decorator: Add operator implementation."""
        def decorator(func):
            cls.operator_registry.add_implementation(func, name, signature)
            return func
        return decorator

    #### Expressions ####

    class LambdaTranslator(Translator):
        def _translate(self, expr):
            # Resolve lambda and return Column object
            if isinstance(expr, LambdaColumn):
                if expr.name not in self.backend.named_cols.fwd:
                    raise ValueError(f"Invalid lambda column '{expr.name}. No column with this name found for table '{self.backend.named_cols}'.'")
                uuid = self.backend.named_cols.fwd[expr.name]
                dtype = self.backend.col_dtype.get(uuid)

                return Column(
                    name = expr.name,
                    table = self.backend,
                    dtype = dtype,
                    uuid = uuid
                )
            return expr

#### ARITHMETIC OPERATORS ######################################################


@AbstractTableImpl.op('__add__', 'int, int -> int')
@AbstractTableImpl.op('__add__', 'int, float -> float')
@AbstractTableImpl.op('__add__', 'float, int -> float')
@AbstractTableImpl.op('__add__', 'float, float -> float')
@AbstractTableImpl.op('__radd__', 'int, int -> int')
@AbstractTableImpl.op('__radd__', 'int, float -> float')
@AbstractTableImpl.op('__radd__', 'float, int -> float')
@AbstractTableImpl.op('__radd__', 'float, float -> float')
def _add(x, y):
    return x + y

@AbstractTableImpl.op('__sub__', 'int, int -> int')
@AbstractTableImpl.op('__sub__', 'int, float -> float')
@AbstractTableImpl.op('__sub__', 'float, int -> float')
@AbstractTableImpl.op('__sub__', 'float, float -> float')
def _sub(x, y):
    return x - y

@AbstractTableImpl.op('__rsub__', 'int, int -> int')
@AbstractTableImpl.op('__rsub__', 'int, float -> float')
@AbstractTableImpl.op('__rsub__', 'float, int -> float')
@AbstractTableImpl.op('__rsub__', 'float, float -> float')
def _rsub(x, y):
    return y - x

@AbstractTableImpl.op('__mul__', 'int, int -> int')
@AbstractTableImpl.op('__mul__', 'int, float -> float')
@AbstractTableImpl.op('__mul__', 'float, int -> float')
@AbstractTableImpl.op('__mul__', 'float, float -> float')
@AbstractTableImpl.op('__rmul__', 'int, int -> int')
@AbstractTableImpl.op('__rmul__', 'int, float -> float')
@AbstractTableImpl.op('__rmul__', 'float, int -> float')
@AbstractTableImpl.op('__rmul__', 'float, float -> float')
def _mul(x, y):
    return x * y

@AbstractTableImpl.op('__truediv__', 'int, int -> float')
@AbstractTableImpl.op('__truediv__', 'int, float -> float')
@AbstractTableImpl.op('__truediv__', 'float, int -> float')
@AbstractTableImpl.op('__truediv__', 'float, float -> float')
def _truediv(x, y):
    return x / y

@AbstractTableImpl.op('__rtruediv__', 'int, int -> float')
@AbstractTableImpl.op('__rtruediv__', 'int, float -> float')
@AbstractTableImpl.op('__rtruediv__', 'float, int -> float')
@AbstractTableImpl.op('__rtruediv__', 'float, float -> float')
def _rtruediv(x, y):
    return y / x

@AbstractTableImpl.op('__floordiv__', 'int, int -> int')
def _floordiv(x, y):
    return x // y

@AbstractTableImpl.op('__rfloordiv__', 'int, int -> int')
def _rfloordiv(x, y):
    return y // x

@AbstractTableImpl.op('__pow__', 'int, int -> int')
def _pow(x, y):
    return x ** y

@AbstractTableImpl.op('__rpow__', 'int, int -> int')
def _rpow(x, y):
    return y ** x

@AbstractTableImpl.op('__mod__', 'int, int -> int')
@AbstractTableImpl.op('__mod__', 'float, int -> float')
def _mod(x, y):
    return x % y

@AbstractTableImpl.op('__rmod__', 'int, int -> int')
@AbstractTableImpl.op('__rmod__', 'int, float -> float')
def _rmod(x, y):
    return y % x

@AbstractTableImpl.op('__neg__', 'int -> int')
@AbstractTableImpl.op('__neg__', 'float -> float')
def _neg(x):
    return -x

@AbstractTableImpl.op('__pos__', 'int -> int')
@AbstractTableImpl.op('__pos__', 'float -> float')
def _pos(x):
    return x


#### BINARY OPERATORS ##########################################################


@AbstractTableImpl.op('__and__', 'bool, bool -> bool')
@AbstractTableImpl.op('__rand__', 'bool, bool -> bool')
def _and(x, y):
    return x & y

@AbstractTableImpl.op('__or__', 'bool, bool -> bool')
@AbstractTableImpl.op('__ror__', 'bool, bool -> bool')
def _or(x, y):
    return x | y

@AbstractTableImpl.op('__xor__', 'bool, bool -> bool')
@AbstractTableImpl.op('__rxor__', 'bool, bool -> bool')
def _xor(x, y):
    return x ^ y

@AbstractTableImpl.op('__invert__', 'bool -> bool')
def _invert(x):
    return ~x


#### COMPARISON OPERATORS ######################################################


@AbstractTableImpl.op('__lt__', 'T, T -> bool')
def _lt(x, y):
    return x < y

@AbstractTableImpl.op('__le__', 'T, T -> bool')
def _le(x, y):
    return x <= y

@AbstractTableImpl.op('__eq__', 'T, U -> bool')
def _eq(x, y):
    return x == y

@AbstractTableImpl.op('__ne__', 'T, U -> bool')
def _ne(x, y):
    return x != y

@AbstractTableImpl.op('__gt__', 'T, T -> bool')
def _gt(x, y):
    return x > y

@AbstractTableImpl.op('__ge__', 'T, T -> bool')
def _ge(x, y):
    return x >= y

