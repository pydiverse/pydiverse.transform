from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable, Sequence

from pydiverse.transform._internal.ops.operator import Operator
from pydiverse.transform._internal.ops.signature import Signature, SignatureTrie
from pydiverse.transform._internal.tree.dtypes import Dtype


class OpImpl:
    """
    Internal type to store the implementation of an operator and all associated
    metadata.

    If the function (`impl`) provided as the underlying parameter has keyword
    only arguments that start with an underscore, they get added to the
    `self.internal_kwargs` list.
    """

    id_ = itertools.count()

    def __init__(
        self,
        operator: Operator,
        impl: Callable,
        signature: Signature,
    ):
        self.operator = operator
        self.impl = impl
        self.signature = signature

        self.__id = next(OpImpl.id_)

        # Inspect impl signature to get internal kwargs
        self.internal_kwargs = self._compute_internal_kwargs(impl)

        # Calculate Ordering Key
        # - Find match with the least number templates in the signature
        # - From those take the one with the least number of different templates
        # - From those take the one where the first template appears latest
        # - From those take the one where the const modifiers match better
        # - From those take the one that isn't a vararg or has the most arguments.
        # - From those take the one that was defined first

        num_templates = 0
        templates_set = set()
        template_indices = []
        const_modifiers = []
        for i, dtype in enumerate(signature.args):
            if isinstance(dtype, dtypes.Template):
                num_templates += 1
                templates_set.add(dtype.name)
                template_indices.append(-i)
            if dtype.const:
                const_modifiers.append(i)
        num_different_templates = len(templates_set)
        is_vararg = int(self.signature.is_vararg)

        self._precedence = (
            num_templates,
            num_different_templates,
            tuple(template_indices),
            tuple(const_modifiers),
            is_vararg,
            -len(signature.args),
            self.__id,
        )

    @staticmethod
    def _compute_internal_kwargs(impl: Callable):
        internal_kwargs = []
        try:
            impl_signature = inspect.signature(impl)
            for name, param in impl_signature.parameters.items():
                if param.kind == inspect.Parameter.KEYWORD_ONLY and name.startswith(
                    "_"
                ):
                    internal_kwargs.append(name)
        except (TypeError, ValueError):
            pass

        return internal_kwargs


class ImplStore:
    def __init__(self, impl_class, super_store=None):
        self.impl_class = impl_class
        self.super_store: ImplStore | None = super_store
        self.implementations: dict[str, SignatureTrie] = dict()

    def add_impl(
        self,
        operator: Operator,
        impl: Callable,
        signature: Signature,
    ):
        impls = self.implementations[operator.name]
        impls.add_impl(OpImpl(operator, impl, signature))

    def get_impl(self, name: str, sig: Sequence[type[Dtype]]) -> Callable:
        assert all(issubclass(t, Dtype) for t in sig)

        if impl := self.implementations.get(name):
            if impl := impl.match(sig):
                return impl

        # If operation hasn't been defined in this registry, go to the parent
        # registry and check if it has been defined there.
        if self.super_store is None:
            raise TypeError(
                f"invalid usage of operator `{name}` with arguments of type "
                + ", ".join(sig)
            )
        return self.super_store.get_impl(name, sig)


class OpRegistrationCM:
    def __init__(self, impl_store: ImplStore, operator: Operator):
        self.impl_store = impl_store
        self.operator = operator

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, *args: type[Dtype]):
        if not isinstance(signature, str):
            raise TypeError("Signature must be of type str.")

        def decorator(func):
            self.impl_store.add_impl(
                self.operator,
                func,
                signature,
            )
            return func

        return decorator

    def auto(self, func: Callable = None):
        if not self.operator.signatures:
            raise ValueError(f"Operator {self.operator} has not default signatures.")

        for signature in self.operator.signatures:
            self.impl_store.add_impl(
                self.operator,
                func,
                signature,
            )

        return func
