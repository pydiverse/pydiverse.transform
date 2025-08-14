# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial, wraps

from pydiverse.transform._internal.errors import SubqueryError
from pydiverse.transform._internal.pipe.cache import Cache
from pydiverse.transform._internal.tree import verbs
from pydiverse.transform._internal.tree.ast import AstNode


class Pipeable:
    def __init__(self, f=None, calls=None):
        if f is not None:
            if calls is not None:
                raise ValueError
            self.calls = [f]
        else:
            self.calls = calls

    def __rshift__(self, other) -> "Pipeable":
        """
        The pipe operator for chaining verbs.
        """

        if isinstance(other, Pipeable):
            return Pipeable(calls=self.calls + other.calls)
        elif callable(other):
            return Pipeable(calls=self.calls + [other])

        raise RuntimeError

    def __call__(self, arg):
        for c in self.calls:
            res = c(arg)
            if isinstance(res, Pipeable):
                res = res(arg)
            arg = res
        return arg


class inverse_partial(partial):
    """
    Just like partial, but the arguments get applied to the back instead of the front.
    This means that a function `def x(a, b, c)` decorated with `@inverse_partial(1, 2)`
    that gets called with `x(0)` is equivalent to calling `x(0, 1, 2)` on the non
    decorated function.
    """

    def __call__(self, /, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.func(*args, *self.args, **keywords)


def verb(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return Pipeable(inverse_partial(fn, *args, **kwargs))

    return wrapper


def check_subquery(new_tbl, cache: Cache, ast_node: AstNode):
    if (reason := cache.requires_subquery(new_tbl._ast)) is not None:
        ast_node.iter_subtree()
        # TODO: we should also search for aliases in the subtree and see if we
        # can make it work by inserting a subquery at one of those.
        if not isinstance(ast_node, verbs.Alias):
            raise SubqueryError(
                f"Executing the `{new_tbl._ast.__class__.__name__.lower()}` verb "
                f"on the table `{ast_node.name}` requires a subquery, which "
                "is forbidden in transform by default.\n"
                f"reason for the subquery: {reason}\n"
                f"hint: Materialize the table `{ast_node.name}` before this "
                "verb. If you are sure you want to do a subquery, put an "
                "`>> alias()` before this verb. "
            )
        return True
    return False


# Checks for subqueries and updates the cache for all verbs except `join`. Since `join`
# is much more complex, we do these tasks manually in the verb.
def modify_ast(fn):
    @wraps(fn)
    def _fn(table, *args, **kwargs):
        new = fn(table, *args, **kwargs)
        assert new._ast != table._ast

        # If a subquery is required, we put a marker in between
        assert new._ast.child == table._ast
        if check_subquery(new, table._cache, table._ast):
            new._ast.child = verbs.SubqueryMarker(new._ast.child)
            new._cache = new._cache.update(new._ast.child)

        new._cache = new._cache.update(new._ast)

        return new

    return _fn
