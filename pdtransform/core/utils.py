class bidict:
    """
    Bidirectional Dictionary
    All keys and values must be unique (bijective one to one mapping).

    To go from key to value use `bidict.fwd`.
    To go from value to key use `bidict.bwd`.
    """

    def __init__(self, seq = None, /, *, fwd = None, bwd = None):
        if fwd is not None and bwd is not None:
            self.__fwd = fwd
            self.__bwd = bwd
        else:
            self.__fwd = dict(seq) if seq is not None else dict()
            self.__bwd = {
                v: k for k, v in self.__fwd.items()
            }

        if len(self.__fwd) != len(self.__bwd) != len(seq):
            raise ValueError(f"Input sequence contains duplicate key value pairs. Mapping must be unique.")

        self.fwd = _BidictInterface(self.__fwd, self.__bwd)
        self.bwd = _BidictInterface(self.__bwd, self.__fwd)

    def __copy__(self):
        return bidict(
            fwd = self.__fwd.copy(),
            bwd = self.__bwd.copy()
        )


class _BidictInterface:
    def __init__(self, fwd: dict, bwd: dict):
        self.__fwd = fwd
        self.__bwd = bwd

    def __setitem__(self, key, value):
        if key in self.__fwd:
            fwd_value = self.__fwd[key]
            del self.__bwd[fwd_value]
        if value in self.__bwd:
            raise ValueError(f"Duplicate value '{value}'. Mapping must be unique.")
        self.__fwd[key] = value
        self.__bwd[value] = key

    def __getitem__(self, key):
        return self.__fwd[key]

    def __delitem__(self, key):
        value = self.__fwd[key]
        del self.__fwd[key]
        del self.__bwd[value]

    def __iter__(self):
        yield from self.__fwd.__iter__()

    def __contains__(self, item):
        return item in self.__fwd

    def items(self):
        return self.__fwd.items()

    def keys(self):
        return self.__fwd.keys()

    def values(self):
        return self.__fwd.values()
