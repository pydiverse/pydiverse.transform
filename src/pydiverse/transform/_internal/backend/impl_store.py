from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Sequence

from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import SignatureTrie
from pydiverse.transform._internal.tree.types import Dtype


class ImplStore:
    __slots__ = ("impl_trie", "default_impl", "impl_manager")

    impl_trie: dict[Operator, SignatureTrie]
    default_impl: dict[Operator, Callable | None]
    impl_manager: ImplContextManager

    def __init__(self) -> None:
        self.impl_trie = dict()
        self.default_impl = dict()
        self.impl_manager = ImplContextManager(self)

    def add_impl(
        self,
        op: Operator,
        sig: Sequence[Dtype] | None,
        is_vararg: bool,
        f: Callable,
    ) -> None:
        if sig is None:
            assert op not in self.default_impl
            self.default_impl[op] = f
        else:
            if op not in self.impl_trie:
                self.impl_trie[op] = SignatureTrie()
            self.impl_trie[op].insert(sig, f, is_vararg)

    def get_impl(self, op: Operator, sig: Sequence[Dtype]) -> Callable | None:
        best_match = None

        if (trie := self.impl_trie.get(op)) is not None:
            trie_match = trie.best_match(sig)
            if trie_match is not None:
                best_match = trie_match[1]
        if best_match is None:
            best_match = self.default_impl.get(op)

        if best_match is None:
            return None

        # filter out only those kwargs that the impl wants
        impl_kwargs = {
            name
            for name, param in inspect.signature(best_match).parameters.items()
            if param.kind == inspect.Parameter.KEYWORD_ONLY
        }

        return lambda *args, **kwargs: best_match(
            *args,
            **{kwarg: val for kwarg, val in kwargs.items() if kwarg in impl_kwargs},
        )


@dataclasses.dataclass(slots=True)
class ImplContextManager:
    impl_store: ImplStore

    def __enter__(self):
        return self

    def __exit__(self, *args): ...

    def __call__(self, op: Operator, *sig: Dtype) -> Callable:
        assert isinstance(op, Operator)

        is_vararg = len(sig) > 1 and sig[-1] is Ellipsis
        if is_vararg:
            sig = sig[:-1]
        assert all(isinstance(dtype, Dtype) for dtype in sig)

        def f(g):
            self.impl_store.add_impl(op, None if len(sig) == 0 else sig, is_vararg, g)
            return g

        return f
