from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic

from pydiverse.transform._typing import ImplT
from pydiverse.transform.core.table_impl import AbstractTableImpl


@dataclass
class JoinDescriptor(Generic[ImplT]):
    __slots__ = ("right", "on", "how")

    right: ImplT
    on: Any
    how: str


class LazyTableImpl(AbstractTableImpl):
    """Base class for lazy backends."""

    pass
