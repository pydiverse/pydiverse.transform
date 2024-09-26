from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from pydiverse.transform import ops
from pydiverse.transform.backend.targets import Target
from pydiverse.transform.ops.core import Ftype
from pydiverse.transform.tree.ast import AstNode
from pydiverse.transform.tree.col_expr import (
    Col,
)
from pydiverse.transform.tree.dtypes import Dtype
from pydiverse.transform.tree.registry import (
    OperatorRegistrationContextManager,
    OperatorRegistry,
)

if TYPE_CHECKING:
    from pydiverse.transform.ops import Operator


class TableImpl(AstNode):
    """
    Base class from which all table backend implementations are derived from.
    """

    registry = OperatorRegistry("TableImpl")

    def __init__(self, name: str, schema: dict[str, Dtype]):
        self.name = name
        self.cols = {
            name: Col(name, self, uuid.uuid1(), dtype, Ftype.EWISE)
            for name, dtype in schema.items()
        }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Add new `registry` class variable to subclass.
        # We define the super registry by walking up the MRO. This allows us
        # to check for potential operation definitions in the parent classes.
        super_reg = None
        for super_cls in cls.__mro__:
            if hasattr(super_cls, "registry"):
                super_reg = super_cls.registry
                break
        cls.registry = OperatorRegistry(cls.__name__, super_reg)

    def iter_subtree(self) -> Iterable[AstNode]:
        yield self

    @staticmethod
    def build_query(nd: AstNode, final_select: list[Col]) -> str | None: ...

    @staticmethod
    def export(nd: AstNode, target: Target, final_select: list[Col]) -> Any: ...

    @classmethod
    def _html_repr_expr(cls, expr):
        """
        Return an appropriate string to display an expression from this backend.
        This is mainly used to IPython.
        """
        return repr(expr)

    @classmethod
    def op(cls, operator: Operator, **kwargs) -> OperatorRegistrationContextManager:
        return OperatorRegistrationContextManager(cls.registry, operator, **kwargs)


with TableImpl.op(ops.NullsFirst()) as op:

    @op.auto
    def _nulls_first(_):
        raise AssertionError


with TableImpl.op(ops.NullsLast()) as op:

    @op.auto
    def _nulls_last(_):
        raise AssertionError


with TableImpl.op(ops.Ascending()) as op:

    @op.auto
    def _ascending(_):
        raise AssertionError


with TableImpl.op(ops.Descending()) as op:

    @op.auto
    def _descending(_):
        raise AssertionError


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
