from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from types import EllipsisType

from pydiverse.transform._internal.ops.ops.aggregation import Any
from pydiverse.transform._internal.tree import types
from pydiverse.transform._internal.tree.types import Dtype


@dataclasses.dataclass(slots=True, init=False)
class Signature:
    types: list[Dtype]
    return_type: Dtype
    is_vararg: bool = False

    def __init__(self, *types: Dtype | EllipsisType, return_type: Dtype):
        self.is_vararg = len(types) >= 1 and types[-1] is Ellipsis
        if self.is_vararg:
            types = types[:-1]
        assert all(isinstance(param, Dtype) for param in types)
        self.types = types
        self.return_type = return_type


# This thing does NOT deal with type template parameters. You have to manually
# instantiate them yourself and insert every one separately.
@dataclasses.dataclass(slots=True)
class SignatureTrie:
    @dataclasses.dataclass(slots=True)
    class Node:
        children: dict[Dtype, SignatureTrie.Node] = dataclasses.field(
            default_factory=dict
        )
        data: Any = None

        def insert(
            self,
            sig: Sequence[Dtype],
            data: Any,
            last_is_vararg: bool,
            *,
            last_type: Dtype | None = None,
        ) -> None:
            if len(sig) == 0:
                assert self.data is None
                self.data = data
                if last_is_vararg:
                    assert isinstance(last_type, Dtype)
                    self.children[last_type] = self
                return

            if sig[0] not in self.children:
                self.children[sig[0]] = SignatureTrie.Node()
            self.children[sig[0]].insert(
                sig[1:], data, last_is_vararg, last_type=sig[0]
            )

        def all_matches(self, sig: Sequence[Dtype]) -> list[tuple[list[Dtype], Any]]:
            if len(sig) == 0:
                return [([], self.data)]

            matches = []
            for dtype, child in self.children.items():
                if sig[0].converts_to(dtype):
                    matches.extend(
                        ([dtype] + match_sig, data)
                        for match_sig, data in child.all_matches(sig[1:])
                    )

            return matches

    root: Node = dataclasses.field(default_factory=Node)

    def insert(self, sig: Sequence[Dtype], data: Any, last_is_vararg: bool) -> None:
        self.root.insert(sig, data, last_is_vararg)

    def best_match(self, sig: Sequence[Dtype]) -> tuple[list[Dtype], Any] | None:
        all_matches = self.root.all_matches(sig)
        if len(all_matches) == 0:
            return None

        return all_matches[
            best_signature_match(sig, [match[0] for match in all_matches])
        ]


# retunrs the index of the signature in `candidates` that matches best
def best_signature_match(
    sig: Sequence[Dtype], candidates: Sequence[Sequence[Dtype]]
) -> int:
    assert len(candidates) > 0

    best = candidates[0]
    best_distance = sig_distance(sig, candidates[0])

    for match in candidates[1:]:
        if best_distance > (this_distance := sig_distance(sig, match)):
            best = match
            best_distance = this_distance

    assert (
        sum(int(best_distance == sig_distance(match, sig)) for match in candidates) == 1
    )
    return best


def sig_distance(sig: Sequence[Dtype], target: Sequence[Dtype]) -> tuple[int, int]:
    return (
        sum(z)
        for z in zip(
            *(types.conversion_cost(s, t) for s, t in zip(sig, target, strict=True)),
            strict=True,
        )
    )
