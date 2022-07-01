import copy
import dataclasses
import uuid
import warnings
from typing import Generic, Any, TypeVar, Iterable, Callable, TYPE_CHECKING

from .column import Column, LambdaColumn, LiteralColumn
from .expressions.operator_registry import OperatorRegistry, TypedOperatorImpl
from .expressions.translator import Translator, DelegatingTranslator, TypedValue
from .utils import bidict, ordered_set


if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from pdtransform.core.expressions import SymbolicExpression


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


ImplT = TypeVar('ImplT', bound = 'AbstractTableImpl')
ExprCompT = TypeVar('ExprCompT', bound = 'TypedValue')
AlignedT = TypeVar('AlignedT', bound = 'TypedValue')
class AbstractTableImpl(metaclass=_TableImplMeta):
    """
    Base class from which all table backend implementations are derived from.
    It tracks various metadata that is relevant for all backends.

    Attributes:
        name: The name of the table.

        selects: Ordered set of selected names.
        named_cols: Map from name to column uuid containing all columns that
            have been named.
        available_cols: Set of UUIDs that can be referenced in symbolic
            expressions. This set gets used to validate verb inputs. It usually
            contains the same uuids as the col_exprs. Only a summarising
            operation resets this.
        col_expr: Map from uuid to the `SymbolicExpression` that corresponds
            to this column.
        col_dtype: Map from uuid to the datatype of the corresponding column.
            It is the responsibility of the backend to keep track of
            this information.

        grouped_by: Ordered set of columns by which the table is grouped by.
        intrinsic_grouped_by: Ordered set of columns representing the underlying
            grouping level of the table. This gets set when performing a
            summarising operation.
    """

    operator_registry: OperatorRegistry

    def __init__(
            self,
            name: str,
            columns: dict[str, Column],
        ):

        self.name = name
        self.compiler = self.ExpressionCompiler(self)
        self.lambda_translator = self.LambdaTranslator(self)

        self.selects = ordered_set()       # type: ordered_set[str]
        self.named_cols = bidict()         # type: bidict[str: uuid.UUID]
        self.available_cols = set()        # type: set[uuid.UUID]
        self.cols = {}                     # type: dict[uuid.UUID: ColumnMetaData]

        self.grouped_by = ordered_set()             # type: ordered_set[Column]
        self.intrinsic_grouped_by = ordered_set()   # type: ordered_set[Column]

        # Init Values
        for name, col in columns.items():
            self.selects.add(name)
            self.named_cols.fwd[name] = col.uuid
            self.available_cols.add(col.uuid)
            self.cols[col.uuid] = ColumnMetaData.from_expr(col.uuid, col, self)

    def copy(self):
        c = copy.copy(self)
        # Copy containers
        for k, v in self.__dict__.items():
            if isinstance(v, (list, dict, set, bidict, ordered_set)):
                c.__dict__[k] = copy.copy(v)

        # Must create a new translator, so that it can access the current df.
        c.compiler = self.ExpressionCompiler(c)
        c.lambda_translator = self.LambdaTranslator(c)
        return c

    def get_col(self, key: str | Column | LambdaColumn):
        """Getter used by `Table.__getattr__`"""

        if isinstance(key, LambdaColumn):
            key = key.name

        if isinstance(key, str):
            if uuid := self.named_cols.fwd.get(key, None):
                return self.cols[uuid].as_column(key, self)
            # Must return AttributeError, else `hasattr` doesn't work on Table instances.
            raise AttributeError(f"Table '{self.name}' has not column named '{key}'.")

        if isinstance(key, Column):
            uuid = key.uuid
            if uuid in self.available_cols:
                name = self.named_cols.bwd[uuid]
                return self.cols[uuid].as_column(name, self)
            raise KeyError(f"Table '{self.name}' has no column that matches '{key}'.")

    def selected_cols(self) -> Iterable[tuple[str, uuid.UUID]]:
        for name in self.selects:
            yield (name, self.named_cols.fwd[name])

    def resolve_lambda_cols(self, expr: Any):
        return self.lambda_translator.translate(expr)

    def is_aligned_with(self, col: Column | LiteralColumn) -> bool:
        """Determine if a column is aligned with the table.

        :param col: The column or literal colum against which alignment
            should be checked.
        :return: A boolean indicating if `col` is aligned with self.
        """
        raise NotImplementedError

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

    def pre_mutate(self, **kwargs):
        """Gives the backend a chance to create a subquery"""
        ...

    def mutate(self, **kwargs):
        ...

    def join(self, right, on, how, *, validate=None):
        ...

    def filter(self, *args):
        ...

    def arrange(self, ordering: 'list[tuple[SymbolicExpression, bool]]'):
        ...

    def group_by(self, *args):
        ...

    def ungroup(self, *args):
        ...

    def pre_summarise(self, **kwargs):
        """Gives the backend a chance to create a subquery"""
        ...

    def summarise(self, **kwargs):
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

    class ExpressionCompiler(Generic[ImplT, ExprCompT], DelegatingTranslator[ExprCompT]):
        """
        Class convert an expression into a function that, when provided with
        the appropriate arguments, evaluates the expression.

        The reason we can't just eagerly evaluate the expression is because for
        grouped data we often have to use the split-apply-combine strategy.
        """

        def __init__(self, backend: ImplT):
            self.backend = backend
            super().__init__(backend.operator_registry)

        def _translate_literal(self, expr, **kwargs):
            def literal_func(*args):
                return expr

            if isinstance(expr, int):
                return TypedValue(literal_func, 'int')
            if isinstance(expr, float):
                return TypedValue(literal_func, 'float')
            if isinstance(expr, str):
                return TypedValue(literal_func, 'str')
            if isinstance(expr, bool):
                return TypedValue(literal_func, 'bool')

    class AlignedExpressionEvaluator(Generic[AlignedT], DelegatingTranslator[AlignedT]):
        """
        Used for evaluating an expression in a typical eager style where, as
        long as two columns have the same alignment / length, we can perform
        operations on them without first having to join them.
        """

        def _translate_literal(self, expr, **kwargs):
            if isinstance(expr, int):
                return TypedValue(expr, 'int')
            if isinstance(expr, float):
                return TypedValue(expr, 'float')
            if isinstance(expr, str):
                return TypedValue(expr, 'str')
            if isinstance(expr, bool):
                return TypedValue(expr, 'bool')

    class LambdaTranslator(Translator):
        """
        Translator that takes an expression and replaces all LambdaColumns
        inside it with the corresponding Column instance.
        """

        def __init__(self, backend: ImplT):
            self.backend = backend
            super().__init__()

        def _translate(self, expr, **kwargs):
            # Resolve lambda and return Column object
            if isinstance(expr, LambdaColumn):
                if expr.name not in self.backend.named_cols.fwd:
                    raise ValueError(f"Invalid lambda column '{expr.name}. No column with this name found for table '{self.backend.named_cols}'.'")
                uuid = self.backend.named_cols.fwd[expr.name]
                return self.backend.cols[uuid].as_column(expr.name, self.backend)
            return expr

    #### Helpers ####

    @classmethod
    def _get_func_ftype(
            cls, args, implementation: TypedOperatorImpl,
            override_ftype: str = None, strict=False) -> str:
        """
        Get the ftype based on a function implementation and the arguments.

            s(s) -> s       a(s) -> a       w(s) -> w
            s(a) -> a       a(a) -> Err     w(a) -> w
            s(w) -> w       a(w) -> Err     w(w) -> Err

        If the implementation ftype is incompatible with the arguments, this
        function raises an Exception.
        """

        ftypes = [arg.ftype for arg in args]
        impl_ftype = override_ftype or implementation.ftype

        if impl_ftype == 's':
            if 'w' in ftypes:
                return 'w'
            if 'a' in ftypes:
                return 'a'
            return 's'

        if impl_ftype == 'a':
            if 'w' in ftypes:
                if strict: raise ValueError(f"Can't nest a window function inside an aggregate function ({implementation.name}).")
                else: warnings.warn(f"Nesting a window function inside an aggregate function is not supported by SQL backend.")
            if 'a' in ftypes:
                if strict: raise ValueError(f"Can't nest an aggregate function inside an aggregate function ({implementation.name}).")
                else: warnings.warn(f"Nesting an aggregate function inside an aggregate function is not supported by SQL backend.")
            return 'a'

        if impl_ftype == 'w':
            if 'w' in ftypes:
                if strict: raise ValueError(f"Can't nest a window function inside a window function ({implementation.name}).")
                else: warnings.warn(f"Nesting a window function inside a window function is not supported by SQL backend.")
            return 'w'


@dataclasses.dataclass
class ColumnMetaData:
    uuid: uuid.UUID
    expr: Any
    compiled: Callable[[Any], TypedValue]
    dtype: str
    ftype: str

    @classmethod
    def from_expr(cls, uuid, expr, table: AbstractTableImpl, **kwargs):
        v = table.compiler.translate(expr, **kwargs)
        return cls(
            uuid = uuid,
            expr = expr,
            compiled = v.value,
            dtype = v.dtype,
            ftype = v.ftype,
        )

    def __hash__(self):
        return hash(self.uuid)

    def as_column(self, name, table: AbstractTableImpl):
        return Column(
            name,
            table,
            self.dtype,
            self.uuid
        )


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

