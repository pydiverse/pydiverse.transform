from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pdtransform.core.table_impl import AbstractTableImpl

ImplT = TypeVar('ImplT', bound = AbstractTableImpl)


@dataclass(slots = True)
class JoinDescriptor(Generic[ImplT]):
    right: ImplT
    on: Any
    how: str


class LazyTableImpl(AbstractTableImpl):
    """ Base class for lazy backends. """
    pass
