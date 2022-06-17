import copy
import uuid

from .column import Column
from .expressions import Translator
from .expressions.expression import SymbolicExpression
from .expressions.lambda_column import LambdaColumn
from .expressions.operator_registry import OperatorRegistry
from .utils import bidict


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
        setattr(c, 'operator_registry', OperatorRegistry(super_reg))
        return c


class AbstractTableImpl(metaclass=_TableImplMeta):
    """
    Base class from which all table backend implementations are derived from.
    It tracks various metadata that is relevant for all backends.
    """

    operator_registry: OperatorRegistry

    def __init__(
            self,
            name: str,
            columns: dict[str, Column],
        ):

        self.name = name
        self.columns = columns
        self.available_columns = set(self.columns.values())  # The set of all columns that are accessible to this table. These are the columns that can be used in a symbolic expression.

        # selects: Set of selected names. (This is implemented using a dict with None value to preserve the order)
        # named_cols: Map from name to column uuid containing all columns that have been named.
        # col_expr: Map from uuid to the `SymbolicExpression` that corresponds to this column.
        # col_dtype: Map from uuid to the datatype of the corresponding column. It is the responsibility of the backend to keep track of this information.
        self.selects = {}           # type: dict[str: None]
        self.named_cols = bidict()  # type: bidict[str: uuid.UUID]
        self.col_expr = {}          # type: dict[uuid.UUID: SymbolicExpression]
        self.col_dtype = {}         # type: dict[uuid.UUID: str]

        # Init Values
        for name, col in columns.items():
            self.selects[name] = None
            self.named_cols.fwd[name] = col._uuid
            self.col_expr[col._uuid] = col
            self.col_dtype[col._uuid] = col._dtype

    def copy(self):
        c = copy.copy(self)
        # Copy containers
        for k, v in self.__dict__.items():
            if isinstance(v, (list, dict, set, bidict)):
                c.__dict__[k] = copy.copy(v)

        return c

    def get_col(self, name: str):
        """Getter used by `Table.__getattr__`"""
        if name in self.columns:
            return self.columns[name]
        raise KeyError(f"Table '{self.name}' has not column named '{name}'.")

    def resolve_lambda_cols(self, expr: SymbolicExpression):
        raise NotImplementedError

    #### Verb Callbacks ####

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


#### ARITHMETIC OPERATORS ######################################################


@AbstractTableImpl.register_op('__add__', 'int, int -> int')
@AbstractTableImpl.register_op('__radd__', 'int, int -> int')
def _add(x, y):
    return x + y

@AbstractTableImpl.register_op('__sub__', 'int, int -> int')
def _sub(x, y):
    return x - y

@AbstractTableImpl.register_op('__rsub__', 'int, int -> int')
def _rsub(x, y):
    return y - x

@AbstractTableImpl.register_op('__mul__', 'int, int -> int')
@AbstractTableImpl.register_op('__rmul__', 'int, int -> int')
def _mul(x, y):
    return x * y

@AbstractTableImpl.register_op('__truediv__', 'int, int -> float')
def _truediv(x, y):
    return x / y

@AbstractTableImpl.register_op('__rtruediv__', 'int, int -> float')
def _rtruediv(x, y):
    return y / x

@AbstractTableImpl.register_op('__floordiv__', 'int, int -> int')
def _floordiv(x, y):
    return x // y

@AbstractTableImpl.register_op('__rfloordiv__', 'int, int -> int')
def _rfloordiv(x, y):
    return y // x

@AbstractTableImpl.register_op('__pow__', 'int, int -> int')
def _pow(x, y):
    return x ** y

@AbstractTableImpl.register_op('__rpow__', 'int, int -> int')
def _rpow(x, y):
    return y ** x

@AbstractTableImpl.register_op('__mod__', 'int, int -> int')
def _mod(x, y):
    return x % y

@AbstractTableImpl.register_op('__rmod__', 'int, int -> int')
def _rmod(x, y):
    return y % x

@AbstractTableImpl.register_op('__neg__', 'int -> int')
def _neg(x):
    return -x

@AbstractTableImpl.register_op('__pos__', 'int -> int')
def _pos(x):
    return x


#### BINARY OPERATORS ##########################################################


@AbstractTableImpl.register_op('__and__', 'bool, bool -> bool')
@AbstractTableImpl.register_op('__rand__', 'bool, bool -> bool')
def _and(x, y):
    return x & y

@AbstractTableImpl.register_op('__or__', 'bool, bool -> bool')
@AbstractTableImpl.register_op('__ror__', 'bool, bool -> bool')
def _or(x, y):
    return x | y

@AbstractTableImpl.register_op('__xor__', 'bool, bool -> bool')
@AbstractTableImpl.register_op('__rxor__', 'bool, bool -> bool')
def _xor(x, y):
    return x ^ y

@AbstractTableImpl.register_op('__invert__', 'bool -> bool')
def _invert(x):
    return ~x


#### COMPARISON OPERATORS ######################################################


@AbstractTableImpl.register_op('__lt__', 'T, T -> bool')
def _lt(x, y):
    return x < y

@AbstractTableImpl.register_op('__le__', 'T, T -> bool')
def _le(x, y):
    return x <= y

@AbstractTableImpl.register_op('__eq__', 'T, U -> bool')
def _eq(x, y):
    return x == y

@AbstractTableImpl.register_op('__ne__', 'T, U -> bool')
def _ne(x, y):
    return x != y

@AbstractTableImpl.register_op('__gt__', 'T, T -> bool')
def _gt(x, y):
    return x > y

@AbstractTableImpl.register_op('__ge__', 'T, T -> bool')
def _ge(x, y):
    return x >= y

