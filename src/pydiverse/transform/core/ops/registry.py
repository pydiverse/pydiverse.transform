from __future__ import annotations

import dataclasses
import functools
import inspect
import itertools
import typing
from functools import partial

from pydiverse.transform.core.ops import dtypes

if typing.TYPE_CHECKING:
    from pydiverse.transform.core.ops import Operator, OperatorExtension


class OperatorImpl:
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
        impl: typing.Callable,
        signature: OperatorSignature,
    ):
        self.operator = operator
        self.impl = impl
        self.signature = signature
        self.variants: dict[str, typing.Callable] = {}

        self.__id = next(OperatorImpl.id_)

        # Inspect impl signature to get internal kwargs
        self.internal_kwargs = []

        try:
            impl_signature = inspect.signature(impl)
            for name, param in impl_signature.parameters.items():
                if param.kind == inspect.Parameter.KEYWORD_ONLY and name.startswith(
                    "_"
                ):
                    self.internal_kwargs.append(name)
        except (TypeError, ValueError):
            pass

        # Calculate Ordering Key
        # - Find match with the least number templates in the signature
        # - From those take the one with the least number of different templates
        # - From those take the one where the first template appears latest
        # - From those take the one that isn't a vararg or has the most arguments.
        # - From those take the one that was defined first

        num_templates = 0
        templates_set = set()
        first_template_index = len(signature.args)
        for i, dtype in enumerate(signature.args):
            if isinstance(dtype, dtypes.Template):
                num_templates += 1
                templates_set.add(dtype.name)
                first_template_index = min(first_template_index, i)
        num_different_templates = len(templates_set)
        is_vararg = int(self.signature.is_vararg)

        self._precedence = (
            num_templates,
            num_different_templates,
            -first_template_index,
            is_vararg,
            -len(signature.args),
            self.__id,
        )

    def add_variant(self, name: str, impl: typing.Callable):
        if name in self.variants:
            raise ValueError(
                f"Already added a variant with name '{name}'"
                f" to operator {self.operator}."
            )
        self.variants[name] = impl


@dataclasses.dataclass
class TypedOperatorImpl:
    """
    Operator Implementation with a non-templated return type.
    Unlike `OperatorImpl`, this class is intended to be the return type of
    the OperatorRegistry.
    """

    operator: Operator
    impl: OperatorImpl
    rtype: dtypes.DType

    @classmethod
    def from_operator_impl(cls, impl: OperatorImpl, rtype: dtypes.DType):
        return cls(
            operator=impl.operator,
            impl=impl,
            rtype=rtype,
        )

    def __call__(self, *args, **kwargs):
        return self.impl.impl(*args, **self.__clean_kwargs(kwargs))

    def get_variant(self, name: str) -> typing.Callable | None:
        variant = self.impl.variants.get(name)
        if variant is None:
            return None

        @functools.wraps(variant)
        def variant_wrapper(*args, **kwargs):
            return variant(*args, **self.__clean_kwargs(kwargs))

        return variant_wrapper

    def __clean_kwargs(self, kwargs):
        return {
            k: v
            for k, v in kwargs.items()
            if not k.startswith("_") or k in self.impl.internal_kwargs
        }


class OperatorRegistry:
    # It only makes sense to define some dunder methods.
    # These are the ones which can be registered.
    SUPPORTED_DUNDER = {
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__truediv__",
        "__rtruediv__",
        "__floordiv__",
        "__rfloordiv__",
        "__pow__",
        "__rpow__",
        "__mod__",
        "__rmod__",
        "__round__",
        "__pos__",
        "__neg__",
        "__abs__",
        "__and__",
        "__rand__",
        "__or__",
        "__ror__",
        "__xor__",
        "__rxor__",
        "__invert__",
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
    }

    # Set containing all operator names that have been defined across all registries.
    # Used for __dir__ method of SymbolicExpression
    ALL_REGISTERED_OPS: set[str] = set()

    def __init__(self, name, super_registry=None):
        self.name = name
        self.super_registry = super_registry
        self.registered_ops = set()  # type: set[Operator]
        self.implementations = dict()  # type: dict[str, OperatorImplementationStore]
        self.check_super = dict()  # type: dict[str, bool]

    def register_op(self, operator: Operator, check_super=True):
        """
        :param operator: The operator to register.
        :param check_super: Bool indicating if the super register should be
            checked if no implementation for this operator can be found.
        """

        name = operator.name
        if name.startswith("__") and name.endswith("__"):
            if name not in OperatorRegistry.SUPPORTED_DUNDER:
                raise ValueError(f"Dunder method {name} is not supported.")

        if operator in self.registered_ops:
            raise ValueError(
                f"Operator {operator} ({name}) already registered in this operator"
                f" registry '{self.name}'"
            )
        if name in self.implementations:
            raise ValueError(
                f"Another operator with the name '{name}' has already been registered"
                " in this registry."
            )

        self.implementations[name] = OperatorImplementationStore(operator)
        self.check_super[name] = check_super

        self.registered_ops.add(operator)
        self.ALL_REGISTERED_OPS.add(name)

    def get_operator(self, name: str) -> Operator | None:
        if impl_store := self.implementations.get(name, None):
            return impl_store.operator

        # If operation hasn't been defined in this registry, go to the parent
        # registry and check if it has been defined there.
        if self.super_registry is None or not self.check_super.get(name, True):
            raise ValueError(f"No implementation for operator '{name}' found")
        return self.super_registry.get_operator(name)

    def add_implementation(
        self,
        operator: Operator,
        impl: typing.Callable,
        signature: str,
        variant: str | None = None,
    ):
        if operator not in self.registered_ops:
            raise ValueError(
                f"Operator {operator} ({operator.name}) hasn't been registered in this"
                f" operator registry '{self.name}'"
            )

        signature = OperatorSignature.parse(signature)
        operator.validate_signature(signature)

        implementation_store = self.implementations[operator.name]
        op_impl = OperatorImpl(operator, impl, signature)

        if variant:
            implementation_store.add_variant(variant, op_impl)
        else:
            implementation_store.add_implementation(op_impl)

    def get_implementation(self, name, args_signature) -> TypedOperatorImpl:
        if name not in self.ALL_REGISTERED_OPS:
            raise ValueError(f"No operator named '{name}'.")

        for dtype in args_signature:
            if not isinstance(dtype, dtypes.DType):
                raise TypeError(
                    "Expected elements of `args_signature` to be of type DType,"
                    f" but found element of type {type(dtype).__name__} instead."
                )

        if store := self.implementations.get(name):
            if impl := store.find_best_match(args_signature):
                return impl

        # If operation hasn't been defined in this registry, go to the parent
        # registry and check if it has been defined there.
        if self.super_registry is None or not self.check_super.get(name, True):
            raise ValueError(
                f"No implementation for operator '{name}' found that matches signature"
                f" '{args_signature}'."
            )
        return self.super_registry.get_implementation(name, args_signature)


class OperatorSignature:
    """
    Specification:

        signature ::= arguments "->" rtype
        arguments ::= (modifiers dtype ",")* terminal_arg
        terminal_arg ::= modifiers (dtype | vararg)
        vararg ::= dtype "..."
        rtype ::= dtype
        dtype ::= template | "int" | "float" | "str" | "bool" | and others...
        modifiers ::= "const"?
        template ::= single uppercase character

    Examples:

        Function that takes two integers and returns an integer:
            int, int -> int

        Templated argument (templates consist of single uppercase characters):
            T, T -> T
            T, U -> bool

        Variable number of arguments:
            int... -> int

    """

    def __init__(self, args: list[dtypes.DType], rtype: dtypes.DType):
        """
        :param args: Tuple of argument types.
        :param rtype: The return type.
        """
        self.args = args
        self.rtype = rtype

    @classmethod
    def parse(cls, signature: str) -> OperatorSignature:
        def parse_cstypes(cst: str):
            # cstypes = comma seperated types
            types = cst.split(",")
            types = [t.strip() for t in types]
            types = [dtypes.dtype_from_string(t) for t in types if t]
            return types

        if "->" not in signature:
            raise ValueError("Invalid signature: arrow (->) missing.")

        arg_sig, r_sig = signature.split("->")
        args = parse_cstypes(arg_sig)
        rtype = parse_cstypes(r_sig)

        # Validate Signature
        if len(rtype) != 1:
            raise ValueError(
                f"Invalid operator signature '{signature}'. Expected exactly one return"
                " type."
            )

        rtype = rtype[0]

        # Validate Template
        # Output template must also occur in signature
        if isinstance(rtype, dtypes.Template):
            if rtype.name not in [arg.name for arg in args]:
                raise ValueError(
                    f"Template return type '{rtype}' must also occur in the"
                    " argument signature."
                )

        # Validate vararg
        # Vararg can only occur for the very last element
        for arg in args[:-1]:
            if arg.vararg:
                raise ValueError("Only last argument can be a vararg.")

        if rtype.vararg:
            raise ValueError("Return value can't be a vararg.")

        return OperatorSignature(
            args=args,
            rtype=rtype,
        )

    def __repr__(self):
        args_str = ", ".join(str(arg) for arg in self.args)
        return args_str + " -> " + str(self.rtype)

    def __hash__(self):
        return hash((self.args, self.rtype))

    def __eq__(self, other):
        if not isinstance(other, OperatorSignature):
            return False
        return self.args == other.args and self.rtype == other.rtype

    @property
    def is_vararg(self) -> bool:
        if len(self.args) == 0:
            return False
        return self.args[-1].vararg


class OperatorImplementationStore:
    """
    Stores all implementations for a specific operation in a trie according to
    their signature. This enables us to easily find the best matching
    operator implementation for a given set of input argument types.
    """

    @dataclasses.dataclass
    class TrieNode:
        __slots__ = ("value", "operator", "children")
        value: dtypes.DType
        operator: OperatorImpl | None
        children: list[OperatorImplementationStore.TrieNode]

    def __init__(self, operator: Operator):
        self.operator = operator
        self.root = self.TrieNode("ROOT", None, [])

    def add_implementation(self, operator: OperatorImpl):
        node = self.get_node(operator.signature, create_missing=True)
        if node.operator is not None:
            raise ValueError(
                f"Implementation for signature {operator.signature} already defined."
            )

        node.operator = operator

    def add_variant(self, name: str, operator: OperatorImpl):
        node = self.get_node(operator.signature, create_missing=False)
        if node is None or node.operator is None:
            raise ValueError(
                f"No implementation for signature {operator.signature} found."
                " Make sure there is an exact match to add a variant."
            )

        assert node.operator.signature.rtype.same_kind(operator.signature.rtype)
        node.operator.add_variant(name, operator.impl)

    def get_node(self, signature: OperatorSignature, create_missing: bool = True):
        node = self.root
        for dtype in signature.args:
            for child in node.children:
                if child.value == dtype:
                    node = child
                    break
            else:
                if create_missing:
                    new_node = self.TrieNode(dtype, None, [])
                    node.children.append(new_node)
                    node = new_node
                else:
                    return None

        return node

    def find_best_match(
        self, signature: tuple[dtypes.DType]
    ) -> TypedOperatorImpl | None:
        matches = list(self._find_matches(signature))

        if not matches:
            return None

        # Find best matching template.
        best_match = None
        best_score = (0xFFFF,)

        for match, templates in matches:
            if match.operator._precedence < best_score:
                best_match = (match, templates)
                best_score = match.operator._precedence

        rtype = best_match[0].operator.signature.rtype
        if isinstance(rtype, dtypes.Template):
            # If rtype is a template -> Translate
            rtype = best_match[1][rtype.name]

        return TypedOperatorImpl.from_operator_impl(best_match[0].operator, rtype)

    def _find_matches(
        self, signature: list[dtypes.DType]
    ) -> typing.Iterable[TrieNode, dict[str, dtypes.DType]]:
        """Yield all operators that match the input signature"""

        # Case 0 arguments:
        if len(signature) == 0:
            yield self.root, dict()
            return

        # Case 1+ args:
        def does_match(
            dtype: dtypes.DType,
            node: OperatorImplementationStore.TrieNode,
            templates: dict[str, dtypes.DType],
        ) -> bool:
            if isinstance(node.value, dtypes.Template):
                t = templates.get(node.value.name)
                return t is None or t.same_kind(dtype)
            return node.value.same_kind(dtype)

        stack = [
            (child, 0, dict()) for child in self.root.children
        ]  # type: list[tuple[OperatorImplementationStore.TrieNode, int, dict]]

        while stack:
            node, s_i, templates = stack.pop()
            dtype = signature[s_i]

            if not does_match(dtype, node, templates):
                continue

            if (
                isinstance(node.value, dtypes.Template)
                and node.value.name not in templates
            ):
                # Insert dtype into templates dict
                templates = templates.copy()
                templates[node.value.name] = dtype

            if s_i + 1 == len(signature):
                if node.operator is not None:
                    yield node, templates
                continue

            children = iter(node.children)
            if node.value.vararg:
                children = itertools.chain(children, iter((node,)))

            for child in children:
                stack.append((child, s_i + 1, templates))


class OperatorRegistrationContextManager:
    def __init__(
        self,
        registry: OperatorRegistry,
        operator: Operator,
        *,
        check_super=True,
    ):
        self.registry = registry
        self.operator = operator

        self.registry.register_op(operator, check_super=check_super)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, signature: str, variant: str = None):
        if not isinstance(signature, str):
            raise TypeError("Signature must be of type str.")

        def decorator(func):
            self.registry.add_implementation(self.operator, func, signature, variant)
            return func

        return decorator

    def auto(self, func=None, *, variant: str = None):
        if func is None:
            return partial(self.auto, variant=variant)

        if not self.operator.signatures:
            raise ValueError(f"Operator {self.operator} has not default signatures.")

        for sig in self.operator.signatures:
            self.registry.add_implementation(self.operator, func, sig, variant)

        return func

    def extension(self, extension: type[OperatorExtension], variant: str = None):
        if extension.operator != type(self.operator):  # noqa: E721
            raise ValueError(
                f"Operator extension for '{extension.operator.__name__}' can't "
                f"be applied to operator of type '{type(self.operator).__name__}'."
            )

        def decorator(func):
            for sig in extension.signatures:
                self.registry.add_implementation(self.operator, func, sig, variant)
            return func

        return decorator
