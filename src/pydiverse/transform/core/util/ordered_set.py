from __future__ import annotations

from typing import Iterable, MutableSet, TypeVar

T = TypeVar("T")


class ordered_set(MutableSet[T]):
    def __init__(self, values: Iterable[T] = tuple()):
        self.__data = {v: None for v in values}

    def __contains__(self, item: T) -> bool:
        return item in self.__data

    def __iter__(self) -> Iterable[T]:
        yield from self.__data.keys()

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self):
        return "{%s}" % ", ".join(repr(e) for e in self)

    def __copy__(self):
        return self.__class__(self)

    def add(self, value: T) -> None:
        self.__data[value] = None

    def discard(self, value: T) -> None:
        del self.__data[value]

    def clear(self) -> None:
        self.__data.clear()

    def copy(self):
        return self.__copy__()

    def pop_back(self) -> None:
        """Return the popped value.Raise KeyError if empty."""
        if len(self) == 0:
            raise KeyError("Ordered set is empty.")
        back = next(reversed(self.__data.keys()))
        self.discard(back)
        return back
