from __future__ import annotations

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.registry import Signature
from pydiverse.transform._internal.tree.types import Bool

bool_and = Operator("__and__", [Signature(Bool(), Bool(), return_type=Bool())])

bool_or = Operator("__or__", [Signature(Bool(), Bool(), return_type=Bool())])

bool_xor = Operator("__xor__", [Signature(Bool(), Bool(), return_type=Bool())])

invert = Operator("__invert__", [Signature(Bool(), return_type=Bool())])
