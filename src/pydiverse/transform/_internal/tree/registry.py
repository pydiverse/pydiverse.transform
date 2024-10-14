from __future__ import annotations

import dataclasses
import functools
import inspect
import itertools
import textwrap
from collections.abc import Callable, Iterable
from functools import partial
from typing import TYPE_CHECKING

from pydiverse.transform._internal.ops.operator import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree import dtypes


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
    def __init__(self, impl_class, super_registry=None):
        self.impl_class = impl_class
        self.super_registry: ImplStore | None = super_registry
        self.implementations: dict[str, SignatureTrie] = dict()
        self.check_super: dict[str, bool] = dict()

    def register_op(self, operator: Operator, check_super=True):
        """
        :param operator: The operator to register.
        :param check_super: Bool indicating if the super register should be
            checked if no implementation for this operator can be found.
        """

        name = operator.name
        if name.startswith("__") and name.endswith("__"):
            if name not in ImplStore.SUPPORTED_DUNDER:
                raise ValueError(f"Dunder method {name} is not supported.")

        if operator in self.registered_ops:
            raise ValueError(
                f"Operator {operator} ({name}) already registered in this operator"
                f" registry '{self.impl_class.__name__}'"
            )
        if name in self.implementations:
            raise ValueError(
                f"Another operator with the name '{name}' has already been registered"
                " in this registry."
            )

        self.implementations[name] = SignatureTrie(operator)
        self.check_super[name] = check_super

        self.registered_ops.add(operator)
        self.ALL_REGISTERED_OPS.add(name)

    def get_op(self, name: str) -> Operator | None:
        if impl_store := self.implementations.get(name, None):
            return impl_store.operator

        # If operation hasn't been defined in this registry, go to the parent
        # registry and check if it has been defined there.
        if self.super_registry is None or not self.check_super.get(name, True):
            raise ValueError(f"no implementation for operator `{name}` found")
        return self.super_registry.get_op(name)

    def add_impl(
        self,
        operator: Operator,
        impl: Callable,
        signature: Signature,
    ):
        impls = self.implementations[operator.name]
        impls.add_impl(OpImpl(operator, impl, signature))

    def get_impl(self, name, args_signature) -> TypedOperatorImpl:
        if name not in self.ALL_REGISTERED_OPS:
            raise ValueError(f"operator named `{name}` does not exist")

        for dtype in args_signature:
            if not isinstance(dtype, dtypes.Dtype):
                raise TypeError(
                    "expected elements of `args_signature` to be of type Dtype, "
                    f"found element of type {type(dtype).__name__} instead"
                )

        if store := self.implementations.get(name):
            if impl := store.find_best_match(args_signature):
                return impl

        # If operation hasn't been defined in this registry, go to the parent
        # registry and check if it has been defined there.
        if self.super_registry is None or not self.check_super.get(name, True):
            raise TypeError(
                f"invalid usage of operator `{name}` with arguments of type "
                f"{args_signature}"
            )
        return self.super_registry.get_impl(name, args_signature)


class OpRegistrationCM:
    def __init__(self, registry: ImplStore, operator: Operator):
        self.registry = registry
        self.operator = operator

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, signature: str):
        if not isinstance(signature, str):
            raise TypeError("Signature must be of type str.")

        def decorator(func):
            self.registry.add_impl(
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
            self.registry.add_impl(
                self.operator,
                func,
                signature,
            )

        return func
