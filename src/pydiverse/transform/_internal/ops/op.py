from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import Any

from pydiverse.transform._internal.ops.signature import Signature, SignatureTrie
from pydiverse.transform._internal.tree.types import Dtype


class Ftype(enum.IntEnum):
    ELEMENT_WISE = 1
    AGGREGATE = 2
    WINDOW = 3


class Operator:
    __slots__ = (
        "name",
        "trie",
        "signatures",
        "ftype",
        "context_kwargs",
        "param_names",
        "default_values",
        "generate_expr_method",
    )

    name: str
    trie: SignatureTrie
    signatures: list[Signature]
    ftype: Ftype
    context_kwargs: list[str]
    param_names: list[str]
    default_values: list[str] | None
    generate_expr_method: bool

    def __init__(
        self,
        name: str,
        *signatures: Signature,
        ftype: Ftype = Ftype.ELEMENT_WISE,
        context_kwargs: list[str] | None = None,
        param_names: list[str] | None = None,
        default_values: list[Any] | None = None,
        generate_expr_method: bool = True,
    ):
        self.name = name
        self.ftype = ftype
        self.context_kwargs = context_kwargs if context_kwargs is not None else []
        self.generate_expr_method = generate_expr_method

        assert all(isinstance(sig, Signature) for sig in signatures)
        self.signatures = signatures
        self.trie = SignatureTrie()
        assert len(signatures) > 0
        for sig in signatures:
            self.trie.insert(sig.types, sig.return_type, sig.is_vararg)

        if param_names is None:
            num_params = len(signatures[0].types)
            assert all(len(sig.types) == num_params for sig in signatures)
            assert num_params <= 2
            if num_params == 1:
                param_names = ["self"]
            elif num_params == 2:
                param_names = ["self", "rhs"]
            else:
                param_names = []

        self.param_names = param_names
        self.default_values = default_values

    def return_type(self, signature: Sequence[Dtype]) -> Dtype:
        match = self.trie.best_match(signature)
        if match is None:
            raise TypeError(
                f"operator `{self.name}` cannot be called with arguments of type "
                f"{", ".join(str(t) for t in signature)}"
            )
        return match[1]
