import dataclasses
import itertools
import typing
import warnings


class OperatorImpl:
    """
    Internal type to store the implementation of an operator and all associated
    metadata.
    """

    id = itertools.count()
    def __init__(self, func, name, signature):
        self.func = func
        self.name = name
        self.signature = signature
        self.rtype = signature.rtype

        self.__id = next(OperatorImpl.id)

        # Calculate Ordering Key
        # - Find match with the least number templates in the signature
        # - From those take the one with the least number of different templates
        # - From those take the one where the first template appears latest
        # - From those take the one that isn't a vararg or has the most number of arguments.
        # - From those take the one that was defined first

        num_templates = 0
        templates_set = set()
        first_template_index = len(signature.args)
        for i, dtype in enumerate(signature.args):
            if is_template_type(dtype):
                num_templates += 1
                templates_set.add(dtype)
                first_template_index = min(first_template_index, i)
        num_different_templates = len(templates_set)
        is_vararg = 1 if is_vararg_type(self.signature.args[-1]) else 0

        self._precedence = (num_templates, num_different_templates, -first_template_index, is_vararg, -len(signature.args), self.__id)


@dataclasses.dataclass
class TypedOperatorImpl:
    """
    Operator Implementation with a non-templated return type.
    Unlike `OperatorImpl`, this class is intended to be the return type of
    the OperatorRegistry.
    """
    name: str
    func: typing.Callable
    rtype: str

    @classmethod
    def from_operator_impl(cls, impl: OperatorImpl, rtype: str):
        return cls(
            name = impl.name,
            func = impl.func,
            rtype = rtype
        )

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class OperatorRegistry:

    # It only makes sense to define some dunder methods.
    # These are the ones which can be registered.
    SUPPORTED_DUNDER = {
        '__add__', '__radd__',
        '__sub__', '__rsub__',
        '__mul__', '__rmul__',
        '__truediv__', '__rtruediv__',
        '__floordiv__', '__rfloordiv__',
        '__pow__', '__rpow__',
        '__mod__', '__rmod__',

        '__round__',
        '__pos__', '__neg__',

        '__and__', '__rand__',
        '__or__', '__ror__',
        '__xor__', '__rxor__',
        '__invert__',

        '__lt__', '__le__',
        '__eq__', '__ne__',
        '__gt__', '__ge__',
    }

    # Set containing all operators that have been defined across all registries.
    ALL_REGISTERED_OPS = set()  # type: set[str]

    def __init__(self, name, super_registry = None):
        self.name = name
        self.super_registry = super_registry
        self.implementations = dict()  # type: dict[str, 'OperationImplementationStore']
        self.check_super = dict()  # type: dict[str, bool]

    def register_op(self, name: str, check_super: bool = False):
        """
        :param name: Name of the new operator.
        :param check_super: Bool indicating if the super register should be
            checked if no implementation for this operator can be found.
        """
        if name.startswith('__') and name.endswith('__'):
            if name not in OperatorRegistry.SUPPORTED_DUNDER:
                raise ValueError(f"Dunder method {name} is not supported.")

        if name in self.implementations:
            warnings.warn(f"Operator '{name}' already registered in operator registry '{self.name}'. All previous implementation will be deleted.")

        self.implementations[name] = OperatorImplementationStore(name)
        self.check_super[name] = check_super
        self.ALL_REGISTERED_OPS.add(name)

    def add_implementation(self, func, name, signature):
        if name not in self.implementations:
            # If not registered before, register with super inheritance.
            self.register_op(name, check_super = True)

        signature = OperatorSignature.parse(signature)
        implementation_store = self.implementations[name]
        implementation_store.add_implementation(OperatorImpl(func, name, signature))

    def get_implementation(self, name, args_signature) -> TypedOperatorImpl:
        if store := self.implementations.get(name):
            if impl := store.find_best_match(args_signature):
                return impl

        # If operation hasn't been defined in this registry, go to the parent
        # registry and check if it has been defined there.
        if self.super_registry is None or not self.check_super.get(name, True):
            raise ValueError(f"No implementation for operator '{name}' found that matches signature '{args_signature}'.")
        return self.super_registry.get_implementation(name, args_signature)


class OperatorSignature:
    """
    Specification:

        signature ::= arguments "->" rtype
        arguments :: (dtype ",")* terminal_arg
        terminal_arg ::= dtype | vararg
        vararg ::= dtype "..."
        rtype ::= dtype
        dtype ::= template | "int" | "float" | "str" | "bool" | and others...
        template ::= single uppercase character

    Examples:

        Function that takes two integers and returns an integer:
            int, int -> int

        Templated argument (templates consist of single uppercase characters):
            T, T -> T
            T, U -> bool

        Variable number of arguments:
            int... -> int

    TODO: Extend syntax.
        - Nullability:         int? -> bool
    """

    def __init__(self, args: tuple[str], rtype: str):
        assert(len(args) > 0)
        self.args = args
        self.rtype = rtype

    @classmethod
    def parse(cls, signature: str) -> 'OperatorSignature':
        def parse_cstypes(cst: str):
            # cstypes = comma seperated types
            types = cst.split(',')
            types = [t.strip() for t in types]
            types = [t for t in types if t]
            return types

        arg_sig, r_sig = signature.split('->')
        args = parse_cstypes(arg_sig)
        rtype = parse_cstypes(r_sig)

        # Validate Signature
        if len(args) == 0:
            raise ValueError(f"Invalid operator signature '{signature}'. No arguments found.")
        if len(rtype) != 1:
            raise ValueError(f"Invalid operator signature '{signature}'. Expected exactly one return type.")

        rtype = rtype[0]

        base_args_types = [get_base_type(arg) for arg in args]
        base_rtype = get_base_type(rtype)

        for type in (*base_args_types, base_rtype):
            if not type.isalnum():
                raise ValueError(f"Invalid type '{type}'. Types must be alphanumeric.")

        # Validate Template
        # Output template must also occur in signature

        if is_template_type(base_rtype):
            if base_rtype not in base_args_types:
                raise ValueError(f"Template return type '{base_rtype}' must also occur in the argument signature.")

        # Validate vararg
        # Vararg can only occur for the very last element

        for arg in args[:-1]:
            if arg.endswith('...'):
                raise ValueError(f"Only last argument can be a vararg.")

        if rtype.endswith('...'):
            raise ValueError(f"Return value can't be a vararg.")

        return OperatorSignature(
            args = tuple(args),
            rtype = rtype
        )

    def __repr__(self):
        return f"{ ', '.join(self.args)} -> {self.rtype}"

    def __hash__(self):
        return hash((self.args, self.rtype))

    def __eq__(self, other):
        if not isinstance(other, OperatorSignature):
            return False
        return self.args == other.args and self.rtype == other.rtype


def is_template_type(s: str) -> bool:
    # Singe upper case ASCII character
    return len(s) == 1 and 65 <= ord(s) <= 90

def is_vararg_type(s: str) -> bool:
    return s.endswith('...')

def get_base_type(s: str) -> str:
    """Get base type from imput string without any decorators.
    -> Trims varargs ellipsis (...) from input.
    """
    return s.removesuffix('...')


class OperatorImplementationStore:
    """
    Stores all implementations for a specific operation in a trie according to
    their signature. This enables us to easily find the best matching
    operator implementation for a given set of input argument types.
    """

    @dataclasses.dataclass
    class TrieNode:
        __slots__ = ('value', 'operator', 'children', 'is_vararg')
        value: str
        operator: OperatorImpl | None
        children: list['OperatorImplementationStore.TrieNode']
        is_vararg: bool


    def __init__(self, name):
        self.name = name
        self.root = self.TrieNode('ROOT', None, [], False)

    def add_implementation(self, operator: OperatorImpl):
        signature = operator.signature
        node = self.root

        for dtype in signature.args:
            dtype = get_base_type(dtype)
            for child in node.children:
                if dtype == child.value:
                    node = child
                    break
            else:
                new_node = self.TrieNode(dtype, None, [], False)
                node.children.append(new_node)
                node = new_node

        if node.operator is not None:
            raise ValueError(f'Implementation for signature {signature} already defined.')

        node.operator = operator
        node.is_vararg |= is_vararg_type(operator.signature.args[-1])

    def find_best_match(self, signature: tuple[str]) -> TypedOperatorImpl | None:
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
        rtype = best_match[1].get(rtype, rtype)  # If it is a template -> Translate

        return TypedOperatorImpl.from_operator_impl(best_match[0].operator, rtype)

    def _find_matches(self, signature: tuple[str]):
        """ Yield all operators that match the input signature """
        def does_match(dtype: str, node: OperatorImplementationStore.TrieNode, templates: dict[str, str]) -> bool:
            if is_template_type(node.value):
                t = templates.get(node.value)
                return t is None or dtype == t
            return dtype == node.value

        # Store tuple of (Node, index in signature, templates)
        stack = [(child, 0, dict()) for child in self.root.children]

        while stack:
            node, s_i, templates = stack.pop()
            dtype = signature[s_i]

            if not does_match(dtype, node, templates):
                continue

            if is_template_type(node.value) and node.value not in templates:
                # Insert dtype into templates dict
                templates = templates.copy()
                templates[node.value] = dtype

            if s_i + 1 == len(signature):
                if node.operator is not None:
                    yield (node, templates)
                continue

            children = iter(node.children)
            if node.is_vararg:
                children = itertools.chain(children, iter((node, )))

            for child in children:
                stack.append((child, s_i + 1, templates))
