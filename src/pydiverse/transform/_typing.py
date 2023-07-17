from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from pydiverse.transform.core.table_impl import AbstractTableImpl


T = TypeVar("T")
ImplT = TypeVar("ImplT", bound="AbstractTableImpl")
CallableT = TypeVar("CallableT", bound=Callable)
