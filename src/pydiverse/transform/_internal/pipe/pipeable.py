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

        if new._cache.requires_subquery(new._ast):
            if not isinstance(new._ast, verbs.Verb) or not isinstance(
                new._ast.child, verbs.Alias
            ):
                raise SubqueryError(
                    f"Executing the `{new._ast.__class__.__name__.lower()}` verb on "
                    f"the table `{new._ast.name}` requires a subquery, which is "
                    "forbidden in transform by default.\n"
                    "hint: If you are sure you want to do a subquery, put an "
                    "`>> alias()` before this verb."
                )

            new._ast.child.subquery = True

        new._cache = table._cache.update(new._ast)
        return new

    return _fn
