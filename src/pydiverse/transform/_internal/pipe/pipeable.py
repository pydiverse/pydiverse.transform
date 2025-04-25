from __future__ import annotations

from functools import partial, wraps

from pydiverse.transform._internal.errors import SubqueryError
from pydiverse.transform._internal.tree import verbs


class Pipeable:
    def __init__(self, f=None, calls=None):
        if f is not None:
            if calls is not None:
                raise ValueError
            self.calls = [f]
        else:
            self.calls = calls

    def __rshift__(self, other) -> Pipeable:
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


def modify_ast(fn):
    @wraps(fn)
    def _fn(table, *args, **kwargs):
        new = fn(table, *args, **kwargs)
        assert new._ast != table._ast

        def _check_subquery(cache, ast_node):
            if cache.requires_subquery(new._ast):
                if not isinstance(ast_node, verbs.Alias):
                    raise SubqueryError(
                        f"Executing the `{new._ast.__class__.__name__.lower()}` verb "
                        f"on the table `{ast_node.name}` requires a subquery, which "
                        "is forbidden in transform by default.\n"
                        f"hint: Materialize the table `{ast_node.name}` before this "
                        "verb. If you are sure you want to do a subquery, put an "
                        "`>> alias()` before this verb. "
                    )
                return True
            return False

        # If a subquery is required, we put a marker in between
        assert new._ast.child == table._ast
        if _check_subquery(table._cache, table._ast):
            new._ast.child = verbs.SubqueryMarker(new._ast.child)
            new._cache = new._cache.update(new._ast.child)

        if isinstance(new._ast, verbs.Join):
            if _check_subquery(args[0]._cache, args[0]._ast):
                new._ast.right = verbs.SubqueryMarker(new._ast.right)
                right_cache = args[0]._cache.update(new._ast.right)
            else:
                right_cache = args[0]._cache

        new._cache = new._cache.update(
            new._ast,
            right_cache=right_cache if isinstance(new._ast, verbs.Join) else None,
        )

        return new

    return _fn
