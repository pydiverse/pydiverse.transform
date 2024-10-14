from __future__ import annotations

from inspect import isclass

from pydiverse.transform._internal.ops.operator import Operator
from pydiverse.transform._internal.tree.registry import SignatureTrie

from . import classes


class OpRegistry:
    name_to_op: dict[str, type[Operator]] = {}
    name_to_signatures: dict[str, SignatureTrie] = {}

    @staticmethod
    def op_by_name(name: str) -> Operator:
        return OpRegistry.op_by_name[name]


for cls in classes.__dict__.values():
    if isclass(cls) and issubclass(cls, Operator):
        assert cls.name not in OpRegistry.name_to_op
        OpRegistry.name_to_op[cls.name] = cls
