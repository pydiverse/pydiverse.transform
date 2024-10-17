from __future__ import annotations

import dataclasses
import enum

from pydiverse.transform._internal.ops.signature import Signature

__all__ = [
    "Ftype",
    "Operator",
    "Operator",
    "Window",
]


class Ftype(enum.IntEnum):
    ELEMENT_WISE = 1
    AGGREGATE = 2
    WINDOW = 3


@dataclasses.dataclass(slots=True)
class Operator:
    name: str
    signatures: list[Signature]
    ftype: Ftype = Ftype.ELEMENT_WISE
    context_kwargs: list[str] = []


@dataclasses.dataclass(slots=True)
class Aggregation(Operator):
    ftype = Ftype.AGGREGATE
    context_kwargs = ["partition_by", "filter"]


@dataclasses.dataclass(slots=True)
class Window(Operator):
    ftype = Ftype.WINDOW
    context_kwargs = ["partition_by", "arrange"]


class NoExprMethod(Operator): ...
