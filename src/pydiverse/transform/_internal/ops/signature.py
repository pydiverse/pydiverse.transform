from __future__ import annotations

import dataclasses
import itertools
import textwrap
from collections.abc import Iterable, Sequence
from types import EllipsisType
from typing import Any, TypeVar

from pydiverse.transform._internal.ops.operator import Operator
from pydiverse.transform._internal.tree import dtypes
from pydiverse.transform._internal.tree.dtypes import Dtype


class Signature:
    params: list[Dtype]
    vararg: bool  # whether the last parameter is vararg
    return_type: Dtype

    def __init__(self, *args: Dtype | EllipsisType, returns: Dtype):
        self.vararg = len(args) > 0 and args[-1] is ...
        assert all(issubclass(arg, Dtype) for arg in args[: -int(self.vararg)])
        if self.vararg:
            assert len(args) > 1
        self.params = args[: -int(self.vararg)]
        assert issubclass(returns, Dtype)
        assert not issubclass(returns, Const)
        if issubclass(returns, DtypeVar):
            assert returns in self.params
        self.return_type = returns


class SignatureTrie:
    """
    Stores all implementations for a specific operation in a trie according to
    their signature. This enables us to easily find the best matching
    operator implementation for a given set of input argument types.
    """

    @dataclasses.dataclass
    class TrieNode:
        __slots__ = ("value", "operator", "children")
        value: type[Dtype]
        children: list[SignatureTrie.TrieNode]
        sig: Signature
        data: Any

        def __repr__(self):
            self_text = f"({self.value} - {self.operator})"
            if self.children:
                children_text = "\n".join(repr(c) for c in self.children)
                children_text = textwrap.indent(children_text, "  ")
                return self_text + "\n" + children_text
            return self_text

    def __init__(self, op: Operator):
        self.op = op
        self.root = self.TrieNode("root", [])

    def add_node(self, sig: Sequence[type[Dtype]], data: Any):
        node = self.get_node(sig, create_missing=True)
        assert node.data is None
        node.data = data

    def get_node(
        self, sig: Sequence[type[Dtype]], create_missing: bool = True
    ) -> TrieNode:
        node = self.root
        for dtype in sig:
            for child in node.children:
                if child.value == dtype:
                    node = child
                    break
            else:
                assert create_missing
                new_node = self.TrieNode(dtype, [])
                node.children.append(new_node)
                node = new_node

        return node

    def match(self, sig: Sequence[type[Dtype]]) -> Any:
        matches = list(self.all_matches(sig))

        if not matches:
            return None

        # Find best matching template.
        best_match: SignatureTrie.TrieNode | None = None
        best_score = ((0x7FFFFFFF,), (0x7FFFFFFF,))

        for match, type_promotion_indices in matches:
            score = (
                # Prefer operators that didn't need any type promotion
                tuple(-i for i in type_promotion_indices),
                # And then match according to signature
                match.operator._precedence,
            )
            if score < best_score:
                best_match = match
                best_score = score

        assert best_match is not None
        return best_match

    def all_matches(
        self, sig: Sequence[type[Dtype]]
    ) -> Iterable[tuple[TrieNode, tuple[int, ...]]]:
        """Yield all operators that match the input signature"""

        # Case 0 arguments:
        if len(sig) == 0:
            yield self.root, tuple()
            return

        # Case 1+ args:
        def does_match(
            dtype: Dtype,
            node: SignatureTrie.TrieNode,
        ) -> bool:
            if issubclass(node.value, DtypeVar):
                return not (
                    issubclass(node.value, Const) and not issubclass(dtype, Const)
                )
            return dtype.can_promote_to(node.value)

        stack: list[tuple[SignatureTrie.TrieNode, int, dict, tuple[int, ...]]] = [
            (child, 0, dict(), tuple()) for child in self.root.children
        ]

        while stack:
            node, s_i, templates, type_promotion_indices = stack.pop()
            dtype = sig[s_i]

            if not does_match(dtype, node):
                continue

            if issubclass(node.value, DtypeVar):
                templates = templates.copy()
                if node.value.name not in templates:
                    templates[node.value.name] = [dtype.without_modifiers()]
                else:
                    templates[node.value.name] = templates[node.value.name] + [
                        dtype.without_modifiers()
                    ]
            elif not node.value.same_kind(dtype):
                # Needs type promotion
                # This only works when types can be promoted once
                # -> (uint > int64) wouldn't be preferred over (uint > int64 > float64)
                type_promotion_indices = (*type_promotion_indices, s_i)

            if s_i + 1 == len(sig):
                if node.operator is not None:
                    # Find compatible type for templates
                    try:
                        templates = {
                            name: dtypes.promote_dtypes(types_)
                            for name, types_ in templates.items()
                        }
                        yield node, type_promotion_indices
                    except TypeError:
                        print(f"Can't promote: {templates}")
                        pass

                continue

            children = iter(node.children)
            if node.value.vararg:
                children = itertools.chain(children, iter((node,)))

            for child in children:
                stack.append((child, s_i + 1, templates, type_promotion_indices))


class Const: ...


def const(dtype: type[Dtype]) -> type[Dtype]:
    return type(f"Const{dtype.__name__}", (Dtype, Const), {})


class DtypeVar(Dtype): ...


class T(DtypeVar): ...
