from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from pydiverse.transform import ops
from pydiverse.transform.backend.targets import Target
from pydiverse.transform.errors import FunctionTypeError
from pydiverse.transform.ops import OpType
from pydiverse.transform.tree.col_expr import (
    Col,
    LiteralCol,
)
from pydiverse.transform.tree.dtypes import DType
from pydiverse.transform.tree.registry import (
    OperatorRegistrationContextManager,
    OperatorRegistry,
)
from pydiverse.transform.tree.table_expr import TableExpr

if TYPE_CHECKING:
    from pydiverse.transform.ops import Operator


class TableImpl:
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

    @staticmethod
    def build_query(expr: TableExpr) -> str | None: ...

    @staticmethod
    def export(expr: TableExpr, target: Target) -> Any: ...

    def col_names(self) -> list[str]: ...

    def schema(self) -> dict[str, DType]: ...

    def clone(self) -> TableImpl: ...

    def is_aligned_with(self, col: Col | LiteralCol) -> bool:
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

    #### Symbolic Operators ####

    @classmethod
    def op(cls, operator: Operator, **kwargs) -> OperatorRegistrationContextManager:
        return OperatorRegistrationContextManager(
            cls.operator_registry, operator, **kwargs
        )

    #### Helpers ####

    @classmethod
    def _get_op_ftype(
        cls, args, operator: Operator, override_ftype: OpType = None, strict=False
    ) -> OpType:
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

        if op_ftype == OpType.EWISE:
            if OpType.WINDOW in ftypes:
                return OpType.WINDOW
            if OpType.AGGREGATE in ftypes:
                return OpType.AGGREGATE
            return op_ftype

        if op_ftype == OpType.AGGREGATE:
            if OpType.WINDOW in ftypes:
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
            if OpType.AGGREGATE in ftypes:
                raise FunctionTypeError(
                    "Can't nest an aggregate function inside an aggregate function"
                    f" ({operator.name})."
                )
            return op_ftype

        if op_ftype == OpType.WINDOW:
            if OpType.WINDOW in ftypes:
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


#### MARKER OPERATIONS #########################################################


with TableImpl.op(ops.NullsFirst()) as op:

    @op.auto
    def _nulls_first(_):
        raise RuntimeError("This is just a marker that never should get called")


with TableImpl.op(ops.NullsLast()) as op:

    @op.auto
    def _nulls_last(_):
        raise RuntimeError("This is just a marker that never should get called")


#### ARITHMETIC OPERATORS ######################################################


with TableImpl.op(ops.Add()) as op:

    @op.auto
    def _add(lhs, rhs):
        return lhs + rhs

    @op.extension(ops.StrAdd)
    def _str_add(lhs, rhs):
        return lhs + rhs


with TableImpl.op(ops.RAdd()) as op:

    @op.auto
    def _radd(rhs, lhs):
        return lhs + rhs

    @op.extension(ops.StrRAdd)
    def _str_radd(lhs, rhs):
        return lhs + rhs


with TableImpl.op(ops.Sub()) as op:

    @op.auto
    def _sub(lhs, rhs):
        return lhs - rhs


with TableImpl.op(ops.RSub()) as op:

    @op.auto
    def _rsub(rhs, lhs):
        return lhs - rhs


with TableImpl.op(ops.Mul()) as op:

    @op.auto
    def _mul(lhs, rhs):
        return lhs * rhs


with TableImpl.op(ops.RMul()) as op:

    @op.auto
    def _rmul(rhs, lhs):
        return lhs * rhs


with TableImpl.op(ops.TrueDiv()) as op:

    @op.auto
    def _truediv(lhs, rhs):
        return lhs / rhs


with TableImpl.op(ops.RTrueDiv()) as op:

    @op.auto
    def _rtruediv(rhs, lhs):
        return lhs / rhs


with TableImpl.op(ops.FloorDiv()) as op:

    @op.auto
    def _floordiv(lhs, rhs):
        return lhs // rhs


with TableImpl.op(ops.RFloorDiv()) as op:

    @op.auto
    def _rfloordiv(rhs, lhs):
        return lhs // rhs


with TableImpl.op(ops.Pow()) as op:

    @op.auto
    def _pow(lhs, rhs):
        return lhs**rhs


with TableImpl.op(ops.RPow()) as op:

    @op.auto
    def _rpow(rhs, lhs):
        return lhs**rhs


with TableImpl.op(ops.Mod()) as op:

    @op.auto
    def _mod(lhs, rhs):
        return lhs % rhs


with TableImpl.op(ops.RMod()) as op:

    @op.auto
    def _rmod(rhs, lhs):
        return lhs % rhs


with TableImpl.op(ops.Neg()) as op:

    @op.auto
    def _neg(x):
        return -x


with TableImpl.op(ops.Pos()) as op:

    @op.auto
    def _pos(x):
        return +x


with TableImpl.op(ops.Abs()) as op:

    @op.auto
    def _abs(x):
        return abs(x)


#### BINARY OPERATORS ##########################################################


with TableImpl.op(ops.And()) as op:

    @op.auto
    def _and(lhs, rhs):
        return lhs & rhs


with TableImpl.op(ops.RAnd()) as op:

    @op.auto
    def _rand(rhs, lhs):
        return lhs & rhs


with TableImpl.op(ops.Or()) as op:

    @op.auto
    def _or(lhs, rhs):
        return lhs | rhs


with TableImpl.op(ops.ROr()) as op:

    @op.auto
    def _ror(rhs, lhs):
        return lhs | rhs


with TableImpl.op(ops.Xor()) as op:

    @op.auto
    def _xor(lhs, rhs):
        return lhs ^ rhs


with TableImpl.op(ops.RXor()) as op:

    @op.auto
    def _rxor(rhs, lhs):
        return lhs ^ rhs


with TableImpl.op(ops.Invert()) as op:

    @op.auto
    def _invert(x):
        return ~x


#### COMPARISON OPERATORS ######################################################


with TableImpl.op(ops.Equal()) as op:

    @op.auto
    def _eq(lhs, rhs):
        return lhs == rhs


with TableImpl.op(ops.NotEqual()) as op:

    @op.auto
    def _ne(lhs, rhs):
        return lhs != rhs


with TableImpl.op(ops.Less()) as op:

    @op.auto
    def _lt(lhs, rhs):
        return lhs < rhs


with TableImpl.op(ops.LessEqual()) as op:

    @op.auto
    def _le(lhs, rhs):
        return lhs <= rhs


with TableImpl.op(ops.Greater()) as op:

    @op.auto
    def _gt(lhs, rhs):
        return lhs > rhs


with TableImpl.op(ops.GreaterEqual()) as op:

    @op.auto
    def _ge(lhs, rhs):
        return lhs >= rhs
