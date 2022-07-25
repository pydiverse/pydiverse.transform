from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydiverse.transform.core.table_impl import AbstractTableImpl

ImplT = TypeVar("ImplT", bound=AbstractTableImpl)


@dataclass
class JoinDescriptor(Generic[ImplT]):
    __slots__ = ("right", "on", "how")

    right: ImplT
    on: Any
    how: str


class LazyTableImpl(AbstractTableImpl):
    """Base class for lazy backends."""

    pass
