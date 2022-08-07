from __future__ import annotations

import copy
import dataclasses
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterable, TypeVar

from pydiverse.transform.core.ops.registry import (
    OperatorRegistrationContextManager,
    OperatorRegistry,
)
from pydiverse.transform.core.util import bidict, ordered_set

from .column import Column, LambdaColumn, LiteralColumn
from .expressions.translator import DelegatingTranslator, Translator, TypedValue
from .ops import Operator, OPType
from .util import OrderingDescriptor

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from pydiverse.transform.core.expressions import SymbolicExpression


class _TableImplMeta(type):
    """
    Metaclass that adds an appropriate operator_registry attribute to each class
    in the table implementation class hierarchy.
    """

    def __new__(cls, name, bases, attrs, **kwargs):
        c = super().__new__(cls, name, bases, attrs, **kwargs)

        # By using `super` to get the super_registry, we can check for potential
        # operation definitions in order of the MRO.
        super_reg = (
            super(c, c).operator_registry
            if hasattr(super(c, c), "operator_registry")
            else None
        )
        setattr(c, "operator_registry", OperatorRegistry(name, super_reg))
        return c


ImplT = TypeVar("ImplT", bound="AbstractTableImpl")
ExprCompT = TypeVar("ExprCompT", bound="TypedValue")
AlignedT = TypeVar("AlignedT", bound="TypedValue")


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

        self.selects = ordered_set()  # type: ordered_set[str]
        self.named_cols = bidict()  # type: bidict[str: uuid.UUID]
        self.available_cols = set()  # type: set[uuid.UUID]
        self.cols = {}  # type: dict[uuid.UUID: ColumnMetaData]

        self.grouped_by = ordered_set()  # type: ordered_set[Column]
        self.intrinsic_grouped_by = ordered_set()  # type: ordered_set[Column]

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

    def preverb_hook(self, verb: str, *args, **kwargs) -> None:
        """Hook that gets called right after `copy` inside a verb

        This gives the backend a chance to react and modify it's state. This
        can, for example, be used to create a subquery based on specific
        conditions.

        :param verb: The name of the verb
        :param args: The arguments passed to the verb
        :param kwargs: The keyword arguments passed to the verb
        """
        ...

    def alias(self, name=None) -> AbstractTableImpl:
        ...

    def collect(self):
        ...

    def build_query(self):
        ...

    def select(self, *args):
        ...

    def mutate(self, **kwargs):
        ...

    def join(self, right, on, how, *, validate=None):
        ...

    def filter(self, *args):
        ...

    def arrange(self, ordering: list[OrderingDescriptor]):
        ...

    def group_by(self, *args):
        ...

    def ungroup(self, *args):
        ...

    def summarise(self, **kwargs):
        ...

    def slice_head(self, n: int, offset: int):
        ...

    #### Symbolic Operators ####

    @classmethod
    def op(cls, operator: Operator, **kwargs) -> OperatorRegistrationContextManager:
        return OperatorRegistrationContextManager(
            cls.operator_registry, operator, **kwargs
        )

    #### Expressions ####

    class ExpressionCompiler(
        Generic[ImplT, ExprCompT], DelegatingTranslator[ExprCompT]
    ):
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
            def literal_func(*args, **kwargs):
                return expr

            if isinstance(expr, int):
                return TypedValue(literal_func, "int")
            if isinstance(expr, float):
                return TypedValue(literal_func, "float")
            if isinstance(expr, str):
                return TypedValue(literal_func, "str")
            if isinstance(expr, bool):
                return TypedValue(literal_func, "bool")

    class AlignedExpressionEvaluator(Generic[AlignedT], DelegatingTranslator[AlignedT]):
        """
        Used for evaluating an expression in a typical eager style where, as
        long as two columns have the same alignment / length, we can perform
        operations on them without first having to join them.
        """

        def _translate_literal(self, expr, **kwargs):
            if isinstance(expr, int):
                return TypedValue(expr, "int")
            if isinstance(expr, float):
                return TypedValue(expr, "float")
            if isinstance(expr, str):
                return TypedValue(expr, "str")
            if isinstance(expr, bool):
                return TypedValue(expr, "bool")

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
                    raise ValueError(
                        f"Invalid lambda column '{expr.name}. No column with this name"
                        f" found for table '{self.backend.name}'.'"
                    )
                uuid = self.backend.named_cols.fwd[expr.name]
                return self.backend.cols[uuid].as_column(expr.name, self.backend)
            return expr

    #### Helpers ####

    @classmethod
    def _get_op_ftype(
        cls, args, operator: Operator, override_ftype: OPType = None, strict=False
    ) -> OPType:
        """
        Get the ftype based on a function implementation and the arguments.

            e(e) -> e       a(e) -> a       w(e) -> w
            e(a) -> a       a(a) -> Err     w(a) -> w
            e(w) -> w       a(w) -> Err     w(w) -> Err

        If the implementation ftype is incompatible with the arguments, this
        function raises an Exception.
        """

        ftypes = [arg.ftype for arg in args]
        op_ftype = override_ftype or operator.ftype

        if op_ftype == OPType.EWISE:
            if OPType.WINDOW in ftypes:
                return OPType.WINDOW
            if OPType.AGGREGATE in ftypes:
                return OPType.AGGREGATE
            return op_ftype

        if op_ftype == OPType.AGGREGATE:
            if OPType.WINDOW in ftypes:
                if strict:
                    raise ValueError(
                        "Can't nest a window function inside an aggregate function"
                        f" ({operator.name})."
                    )
                else:
                    # TODO: Replace with logger
                    warnings.warn(
                        f"Nesting a window function inside an aggregate function is not"
                        f" supported by SQL backend."
                    )
            if OPType.AGGREGATE in ftypes:
                raise ValueError(
                    "Can't nest an aggregate function inside an aggregate function"
                    f" ({operator.name})."
                )
            return op_ftype

        if op_ftype == OPType.WINDOW:
            if OPType.WINDOW in ftypes:
                if strict:
                    raise ValueError(
                        "Can't nest a window function inside a window function"
                        f" ({operator.name})."
                    )
                else:
                    warnings.warn(
                        f"Nesting a window function inside a window function is not"
                        f" supported by SQL backend."
                    )
            return op_ftype


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
            uuid=uuid,
            expr=expr,
            compiled=v.value,
            dtype=v.dtype,
            ftype=v.ftype,
        )

    def __hash__(self):
        return hash(self.uuid)

    def as_column(self, name, table: AbstractTableImpl):
        return Column(name, table, self.dtype, self.uuid)


#### ARITHMETIC OPERATORS ######################################################


from pydiverse.transform.core import ops

with AbstractTableImpl.op(ops.Add()) as op:

    @op.auto
    def _add(lhs, rhs):
        return lhs + rhs

    @op.extension(ops.StringAdd)
    def _str_add(lhs, rhs):
        return lhs + rhs


with AbstractTableImpl.op(ops.RAdd()) as op:

    @op.auto
    def _radd(rhs, lhs):
        return lhs + rhs

    @op.extension(ops.StringRAdd)
    def _str_radd(lhs, rhs):
        return lhs + rhs


with AbstractTableImpl.op(ops.Sub()) as op:

    @op.auto
    def _sub(lhs, rhs):
        return lhs - rhs


with AbstractTableImpl.op(ops.RSub()) as op:

    @op.auto
    def _rsub(rhs, lhs):
        return lhs - rhs


with AbstractTableImpl.op(ops.Mul()) as op:

    @op.auto
    def _mul(lhs, rhs):
        return lhs * rhs


with AbstractTableImpl.op(ops.RMul()) as op:

    @op.auto
    def _rmul(rhs, lhs):
        return lhs * rhs


with AbstractTableImpl.op(ops.TrueDiv()) as op:

    @op.auto
    def _truediv(lhs, rhs):
        return lhs / rhs


with AbstractTableImpl.op(ops.RTrueDiv()) as op:

    @op.auto
    def _rtruediv(rhs, lhs):
        return lhs / rhs


with AbstractTableImpl.op(ops.FloorDiv()) as op:

    @op.auto
    def _floordiv(lhs, rhs):
        return lhs // rhs


with AbstractTableImpl.op(ops.RFloorDiv()) as op:

    @op.auto
    def _rfloordiv(rhs, lhs):
        return lhs // rhs


with AbstractTableImpl.op(ops.Pow()) as op:

    @op.auto
    def _pow(lhs, rhs):
        return lhs**rhs


with AbstractTableImpl.op(ops.RPow()) as op:

    @op.auto
    def _rpow(rhs, lhs):
        return lhs**rhs


with AbstractTableImpl.op(ops.Mod()) as op:

    @op.auto
    def _mod(lhs, rhs):
        return lhs % rhs


with AbstractTableImpl.op(ops.RMod()) as op:

    @op.auto
    def _rmod(rhs, lhs):
        return lhs % rhs


with AbstractTableImpl.op(ops.Neg()) as op:

    @op.auto
    def _neg(x):
        return -x


with AbstractTableImpl.op(ops.Pos()) as op:

    @op.auto
    def _pos(x):
        return +x


with AbstractTableImpl.op(ops.Abs()) as op:

    @op.auto
    def _abs(x):
        return abs(x)


#### BINARY OPERATORS ##########################################################


with AbstractTableImpl.op(ops.And()) as op:

    @op.auto
    def _and(lhs, rhs):
        return lhs & rhs


with AbstractTableImpl.op(ops.RAnd()) as op:

    @op.auto
    def _rand(rhs, lhs):
        return lhs & rhs


with AbstractTableImpl.op(ops.Or()) as op:

    @op.auto
    def _or(lhs, rhs):
        return lhs | rhs


with AbstractTableImpl.op(ops.ROr()) as op:

    @op.auto
    def _ror(rhs, lhs):
        return lhs | rhs


with AbstractTableImpl.op(ops.Xor()) as op:

    @op.auto
    def _xor(lhs, rhs):
        return lhs ^ rhs


with AbstractTableImpl.op(ops.RXor()) as op:

    @op.auto
    def _rxor(rhs, lhs):
        return lhs ^ rhs


with AbstractTableImpl.op(ops.Invert()) as op:

    @op.auto
    def _invert(x):
        return ~x


#### COMPARISON OPERATORS ######################################################


with AbstractTableImpl.op(ops.Equal()) as op:

    @op.auto
    def _eq(lhs, rhs):
        return lhs == rhs


with AbstractTableImpl.op(ops.NotEqual()) as op:

    @op.auto
    def _ne(lhs, rhs):
        return lhs != rhs


with AbstractTableImpl.op(ops.Less()) as op:

    @op.auto
    def _lt(lhs, rhs):
        return lhs < rhs


with AbstractTableImpl.op(ops.LessEqual()) as op:

    @op.auto
    def _le(lhs, rhs):
        return lhs <= rhs


with AbstractTableImpl.op(ops.Greater()) as op:

    @op.auto
    def _gt(lhs, rhs):
        return lhs > rhs


with AbstractTableImpl.op(ops.GreaterEqual()) as op:

    @op.auto
    def _ge(lhs, rhs):
        return lhs >= rhs
