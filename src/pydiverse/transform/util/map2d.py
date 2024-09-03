from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Generic, TypeVar

T = TypeVar("T", bound=Hashable)
U = TypeVar("U")


class Map2d(Generic[T, U]):
    def __init__(self, mapping: dict[T, U] | None = None) -> Map2d[T, U]:
        if mapping is None:
            mapping = dict()
        self.mapping = mapping

    def inner_update(self, other: Map2d | dict):
        mapping = other if isinstance(other, dict) else other.mapping
        for key, val in mapping.items():
            self_val = self.mapping.get(key)
            if self_val:
                self_val.update(val)
            else:
                self[key] = val

    def inner_map(self, fn: Callable[[U], U]):
        self.mapping = {
            outer_key: {inner_key: fn(val) for inner_key, val in inner_map.items()}
            for outer_key, inner_map in self.mapping.items()
        }

    def keys(self):
        return self.mapping.keys()

    def values(self):
        return self.mapping.values()

    def items(self):
        return self.mapping.items()

    def __contains__(self, key):
        return self.mapping.__contains__(key)

    def __iter__(self):
        return self.mapping.__iter__()

    def __setitem__(self, item, value):
        return self.mapping.__setitem__(item, value)

    def __getitem__(self, item):
        return self.mapping.__getitem__(item)

    def __delitem__(self, item):
        return self.mapping.__delitem__(item)
