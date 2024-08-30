from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from pydiverse.transform.core.table_impl import TableImpl


T = TypeVar("T")
ImplT = TypeVar("ImplT", bound="TableImpl")
CallableT = TypeVar("CallableT", bound=Callable)
