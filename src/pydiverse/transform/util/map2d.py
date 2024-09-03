from __future__ import annotations

from collections.abc import Hashable
from typing import Generic, TypeVar

T = TypeVar("T", bound=Hashable)
U = TypeVar("U")


class Map2d(Generic[T, U]):
    def __init__(self, mapping: dict[T, U] | None = None) -> Map2d[T, U]:
        if mapping is None:
            mapping = dict()
        self.mapping = mapping

    def inner_update(self, other: Map2d):
        for key, val in other.mapping:
            self_val = self.mapping.get(key)
            if self_val:
                self_val.update(val)
            else:
                self[key] = val

    def keys(self):
        return self.mapping.keys()

    def values(self):
        return self.mapping.values()

    def items(self):
        return self.mapping.items()

    def __iter__(self):
        return self.mapping.__iter__()

    def __setitem__(self, item, value):
        return self.mapping.__setitem__(item, value)

    def __getitem__(self, item):
        return self.mapping.__getitem__(item)

    def __delitem__(self, item):
        return self.mapping.__delitem__(item)
