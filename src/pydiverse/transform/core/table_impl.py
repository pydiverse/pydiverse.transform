from __future__ import annotations

import copy
import dataclasses
import datetime
import uuid
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from pydiverse.transform import ops
from pydiverse.transform._typing import ImplT
from pydiverse.transform.core import dtypes
from pydiverse.transform.core.expressions import (
    CaseExpression,
    Column,
    LambdaColumn,
    LiteralColumn,
)
from pydiverse.transform.core.expressions.translator import (
    DelegatingTranslator,
    Translator,
    TypedValue,
)
from pydiverse.transform.core.registry import (
    OperatorRegistrationContextManager,
    OperatorRegistry,
)
from pydiverse.transform.core.util import bidict, ordered_set
from pydiverse.transform.errors import ExpressionTypeError, FunctionTypeError
from pydiverse.transform.ops import OPType

if TYPE_CHECKING:
    from pydiverse.transform.core.util import OrderingDescriptor
    from pydiverse.transform.ops import Operator


ExprCompT = TypeVar("ExprCompT", bound="TypedValue")
AlignedT = TypeVar("AlignedT", bound="TypedValue")


class AbstractTableImpl:
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

    operator_registry = OperatorRegistry("AbstractTableImpl")

    def __init__(
        self,
        name: str,
        columns: dict[str, Column],
    ):
        self.name = name
        self.compiler = self.ExpressionCompiler(self)
        self.lambda_translator = self.LambdaTranslator(self)

        self.selects: ordered_set[str] = ordered_set()  # subset of named_cols
        self.named_cols: bidict[str, uuid.UUID] = bidict()
        self.available_cols: set[uuid.UUID] = set()
        self.cols: dict[uuid.UUID, ColumnMetaData] = dict()

        self.grouped_by: ordered_set[Column] = ordered_set()
        self.intrinsic_grouped_by: ordered_set[Column] = ordered_set()

        # Init Values
        for name, col in columns.items():
            self.selects.add(name)
            self.named_cols.fwd[name] = col.uuid
            self.available_cols.add(col.uuid)
            self.cols[col.uuid] = ColumnMetaData.from_expr(col.uuid, col, self)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Add new `operator_registry` class variable to subclass.
        # We define the super registry by walking up the MRO. This allows us
        # to check for potential operation definitions in the parent classes.
        super_reg = None
        for super_cls in cls.__mro__:
            if hasattr(super_cls, "operator_registry"):
                super_reg = super_cls.operator_registry
                break
        cls.operator_registry = OperatorRegistry(cls.__name__, super_reg)

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
            # Must return AttributeError, else `hasattr` doesn't work on Table instances
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

    def alias(self, name=None) -> AbstractTableImpl: ...

    def collect(self): ...

    def build_query(self): ...

    def select(self, *args): ...

    def mutate(self, **kwargs): ...

    def join(self, right, on, how, *, validate="m:m"): ...

    def filter(self, *args): ...

    def arrange(self, ordering: list[OrderingDescriptor]): ...

    def group_by(self, *args): ...

    def ungroup(self, *args): ...

    def summarise(self, **kwargs): ...

    def slice_head(self, n: int, offset: int): ...

    def export(self): ...

    #### Symbolic Operators ####

    @classmethod
    def op(cls, operator: Operator, **kwargs) -> OperatorRegistrationContextManager:
        return OperatorRegistrationContextManager(
            cls.operator_registry, operator, **kwargs
        )

    #### Expressions ####

    class ExpressionCompiler(
        DelegatingTranslator[ExprCompT], Generic[ImplT, ExprCompT]
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
            literal = self._translate_literal_value(expr)

            if isinstance(expr, bool):
                return TypedValue(literal, dtypes.Bool(const=True))
            if isinstance(expr, int):
                return TypedValue(literal, dtypes.Int(const=True))
            if isinstance(expr, float):
                return TypedValue(literal, dtypes.Float(const=True))
            if isinstance(expr, str):
                return TypedValue(literal, dtypes.String(const=True))
            if isinstance(expr, datetime.datetime):
                return TypedValue(literal, dtypes.DateTime(const=True))
            if isinstance(expr, datetime.date):
                return TypedValue(literal, dtypes.Date(const=True))
            if isinstance(expr, datetime.timedelta):
                return TypedValue(literal, dtypes.Duration(const=True))

            if expr is None:
                return TypedValue(literal, dtypes.NoneDType(const=True))

        def _translate_literal_value(self, expr):
            def literal_func(*args, **kwargs):
                return expr

            return literal_func

        def _translate_case_common(
            self,
            expr: CaseExpression,
            switching_on: ExprCompT | None,
            cases: list[tuple[ExprCompT, ExprCompT]],
            default: ExprCompT,
            **kwargs,
        ) -> tuple[dtypes.DType, OPType]:
            # Determine dtype of result
            val_dtypes = [default.dtype.without_modifiers()]
            for _, val in cases:
                val_dtypes.append(val.dtype.without_modifiers())

            result_dtype = dtypes.promote_dtypes(val_dtypes)

            # Determine ftype of result
            val_ftypes = set()
            if not default.dtype.const:
                val_ftypes.add(default.ftype)

            for _, val in cases:
                if not val.dtype.const:
                    val_ftypes.add(val.ftype)

            if len(val_ftypes) == 0:
                result_ftype = OPType.EWISE
            elif len(val_ftypes) == 1:
                (result_ftype,) = val_ftypes
            elif OPType.WINDOW in val_ftypes:
                result_ftype = OPType.WINDOW
            else:
                # AGGREGATE and EWISE are incompatible
                raise FunctionTypeError(
                    "Incompatible function types found in case statement: " ", ".join(
                        val_ftypes
                    )
                )

            if result_ftype is OPType.EWISE and switching_on is not None:
                result_ftype = switching_on.ftype

            # Type check conditions
            if switching_on is None:
                # All conditions must be boolean
                for cond, _ in cases:
                    if not dtypes.Bool().same_kind(cond.dtype):
                        raise ExpressionTypeError(
                            "All conditions in a case statement return booleans. "
                            f"{cond} is of type {cond.dtype}."
                        )
            else:
                # All conditions must be of the same type as switching_on
                for cond, _ in cases:
                    if not cond.dtype.can_promote_to(
                        switching_on.dtype.without_modifiers()
                    ):
                        # Can't compare
                        raise ExpressionTypeError(
                            f"Condition value {cond} (dtype: {cond.dtype}) "
                            f"is incompatible with switch dtype {switching_on.dtype}."
                        )

            return result_dtype, result_ftype

    class AlignedExpressionEvaluator(DelegatingTranslator[AlignedT], Generic[AlignedT]):
        """
        Used for evaluating an expression in a typical eager style where, as
        long as two columns have the same alignment / length, we can perform
        operations on them without first having to join them.
        """

        def _translate_literal(self, expr, **kwargs):
            if isinstance(expr, bool):
                return TypedValue(expr, dtypes.Bool(const=True))
            if isinstance(expr, int):
                return TypedValue(expr, dtypes.Int(const=True))
            if isinstance(expr, float):
                return TypedValue(expr, dtypes.Float(const=True))
            if isinstance(expr, str):
                return TypedValue(expr, dtypes.String(const=True))
            if isinstance(expr, datetime.datetime):
                return TypedValue(expr, dtypes.DateTime(const=True))
            if isinstance(expr, datetime.date):
                return TypedValue(expr, dtypes.Date(const=True))
            if isinstance(expr, datetime.timedelta):
                return TypedValue(expr, dtypes.Duration(const=True))

            if expr is None:
                return TypedValue(expr, dtypes.NoneDType(const=True))

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
                        f"Invalid lambda column '{expr.name}'. No column with this name"
                        f" found for table '{self.backend.name}'."
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
                    raise FunctionTypeError(
                        "Can't nest a window function inside an aggregate function"
                        f" ({operator.name})."
                    )
                else:
                    # TODO: Replace with logger
                    warnings.warn(
                        "Nesting a window function inside an aggregate function is not"
                        " supported by SQL backend."
                    )
            if OPType.AGGREGATE in ftypes:
                raise FunctionTypeError(
                    "Can't nest an aggregate function inside an aggregate function"
                    f" ({operator.name})."
                )
            return op_ftype

        if op_ftype == OPType.WINDOW:
            if OPType.WINDOW in ftypes:
                if strict:
                    raise FunctionTypeError(
                        "Can't nest a window function inside a window function"
                        f" ({operator.name})."
                    )
                else:
                    warnings.warn(
                        "Nesting a window function inside a window function is not"
                        " supported by SQL backend."
                    )
            return op_ftype


@dataclasses.dataclass
class ColumnMetaData:
    uuid: uuid.UUID
    expr: Any
    compiled: Callable[[Any], TypedValue]
    dtype: dtypes.DType
    ftype: OPType

    @classmethod
    def from_expr(cls, uuid, expr, table: AbstractTableImpl, **kwargs):
        v: TypedValue = table.compiler.translate(expr, **kwargs)
        return cls(
            uuid=uuid,
            expr=expr,
            compiled=v.value,
            dtype=v.dtype.without_modifiers(),
            ftype=v.ftype,
        )

    def __hash__(self):
        return hash(self.uuid)

    def as_column(self, name, table: AbstractTableImpl):
        return Column(name, table, self.dtype, self.uuid)


#### MARKER OPERATIONS #########################################################


with AbstractTableImpl.op(ops.NullsFirst()) as op:

    @op.auto
    def _nulls_first(_):
        raise RuntimeError("This is just a marker that never should get called")


with AbstractTableImpl.op(ops.NullsLast()) as op:

    @op.auto
    def _nulls_last(_):
        raise RuntimeError("This is just a marker that never should get called")


#### ARITHMETIC OPERATORS ######################################################


with AbstractTableImpl.op(ops.Add()) as op:

    @op.auto
    def _add(lhs, rhs):
        return lhs + rhs

    @op.extension(ops.StrAdd)
    def _str_add(lhs, rhs):
        return lhs + rhs


with AbstractTableImpl.op(ops.RAdd()) as op:

    @op.auto
    def _radd(rhs, lhs):
        return lhs + rhs

    @op.extension(ops.StrRAdd)
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
