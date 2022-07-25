from __future__ import annotations

from typing import (
    Generic,
    ItemsView,
    Iterable,
    KeysView,
    Mapping,
    MutableMapping,
    TypeVar,
    ValuesView,
)

KT = TypeVar("KT")
VT = TypeVar("VT")


class bidict(Generic[KT, VT]):
    """
    Bidirectional Dictionary
    All keys and values must be unique (bijective one to one mapping).

    To go from key to value use `bidict.fwd`.
    To go from value to key use `bidict.bwd`.
    """

    def __init__(self, seq: Mapping[KT, VT] = None, /, *, fwd=None, bwd=None):
        if fwd is not None and bwd is not None:
            self.__fwd = fwd
            self.__bwd = bwd
        else:
            self.__fwd = dict(seq) if seq is not None else dict()
            self.__bwd = {v: k for k, v in self.__fwd.items()}

        if len(self.__fwd) != len(self.__bwd) != len(seq):
            raise ValueError(
                f"Input sequence contains duplicate key value pairs. Mapping must be"
                f" unique."
            )

        self.fwd = _BidictInterface(
            self.__fwd, self.__bwd
        )  # type: _BidictInterface[KT, VT]
        self.bwd = _BidictInterface(
            self.__bwd, self.__fwd
        )  # type: _BidictInterface[VT, KT]

    def __copy__(self):
        return bidict(fwd=self.__fwd.copy(), bwd=self.__bwd.copy())

    def __len__(self):
        return len(self.__fwd)

    def clear(self):
        self.__fwd.clear()
        self.__bwd.clear()


class _BidictInterface(MutableMapping[KT, VT]):
    def __init__(self, fwd: dict[KT, VT], bwd: dict[VT, KT]):
        self.__fwd = fwd
        self.__bwd = bwd

    def __setitem__(self, key: KT, value: VT):
        if key in self.__fwd:
            fwd_value = self.__fwd[key]
            del self.__bwd[fwd_value]
        if value in self.__bwd:
            raise ValueError(f"Duplicate value '{value}'. Mapping must be unique.")
        self.__fwd[key] = value
        self.__bwd[value] = key

    def __getitem__(self, key: KT) -> VT:
        return self.__fwd[key]

    def __delitem__(self, key: KT):
        value = self.__fwd[key]
        del self.__fwd[key]
        del self.__bwd[value]

    def __iter__(self) -> Iterable[KT]:
        yield from self.__fwd.__iter__()

    def __len__(self) -> int:
        return len(self.__fwd)

    def __contains__(self, item) -> bool:
        return item in self.__fwd

    def items(self) -> ItemsView[KT, VT]:
        return self.__fwd.items()

    def keys(self) -> KeysView[KT]:
        return self.__fwd.keys()

    def values(self) -> ValuesView[VT]:
        return self.__fwd.values()
