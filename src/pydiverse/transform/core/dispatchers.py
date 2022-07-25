from __future__ import annotations

import copy
from functools import partial, reduce, wraps
from typing import Any

from pydiverse.transform.core import column, table, verbs
from pydiverse.transform.core.expressions import unwrap_symbolic_expressions
from pydiverse.transform.core.table_impl import AbstractTableImpl
from pydiverse.transform.core.util import bidict, traverse


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
        Pipeable >> other
        -> Lazy. Extend pipe.
        """
        if isinstance(other, Pipeable):
            return Pipeable(calls=self.calls + other.calls)
        elif callable(other):
            return Pipeable(calls=self.calls + [other])

        raise Exception

    def __rrshift__(self, other):
        """
        other >> Pipeable
        -> Eager.
        """
        if callable(other):
            return Pipeable(calls=[other] + self.calls)
        return self(other)

    def __call__(self, arg):
        return reduce(lambda x, f: f(x), self.calls, arg)


class inverse_partial(partial):
    """
    Just like partial, but the arguments get applied to the back instead of the front.
    This means that a function `def x(a, b, c)` decorated with `@inverse_partial(1, 2)`
    that gets called with `x(0)` is equivalent to calling `x(0, 1, 2)` on the non
    decorated function.
    """

    def __call__(self, /, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        #                   ↙ *args moved to front.
        return self.func(*args, *self.args, **keywords)


def verb(func):
    def copy_tables(arg: Any = None):
        return traverse(
            arg, lambda x: copy.copy(x) if isinstance(x, table.Table) else x
        )

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Copy Table objects to prevent mutating them
        # This can be the case if the user uses __setitem__ inside the verb
        def f(*args, **kwargs):
            args = copy_tables(args)
            kwargs = copy_tables(kwargs)
            return func(*args, **kwargs)

        f = inverse_partial(f, *args, **kwargs)  # Bind arguments
        return Pipeable(f)

    return wrapper


def builtin_verb(backends=None):
    def wrap_and_unwrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            args = list(args)
            args = unwrap_symbolic_expressions(args)
            if len(args):
                args[0] = col_to_table(args[0])
            args = unwrap_tables(args)

            kwargs = unwrap_symbolic_expressions(kwargs)
            kwargs = unwrap_tables(kwargs)

            return wrap_tables(func(*args, **kwargs))

        return wrapper

    def check_backend(func):
        if backends is None:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            assert len(args) > 0
            impl = args[0]._impl
            if isinstance(impl, backends):
                return func(*args, **kwargs)
            raise TypeError(f"Backend {impl} not supported for verb '{func.__name__}'.")

        return wrapper

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            f = func
            f = wrap_and_unwrap(f)  # Convert from Table to Impl and back
            f = check_backend(f)  # Check type of backend
            f = inverse_partial(f, *args, **kwargs)  # Bind arguments
            return Pipeable(f)  # Make pipeable

        return wrapper

    return decorator


# Helper


def col_to_table(arg: Any = None):
    """
    Takes a single argument and if it is a column, replaces it with a table
    implementation that only contains that one column.

    This allows for more eager style code where you perform operations on
    columns like with the following example::

        def get_c(b, tB):
            tC = b >> left_join(tB, b == tB.b)
            return tC[tB.c]
        feature_col = get_c(tblA.b, tblB)

    """

    if isinstance(arg, column.Column):
        tbl = (arg.table >> verbs.select(arg))._impl  # type: AbstractTableImpl
        col = tbl.get_col(arg)

        tbl.available_cols = {col.uuid}
        tbl.named_cols = bidict({col.name: col.uuid})
        return tbl
    elif isinstance(arg, column.LambdaColumn):
        raise ValueError

    return arg


def unwrap_tables(arg: Any = None):
    """
    Takes an instance or collection of `Table` objects and replaces them with
    their implementation.
    """
    return traverse(arg, lambda x: x._impl if isinstance(x, table.Table) else x)


def wrap_tables(arg: Any = None):
    """
    Takes an instance or collection of `AbstractTableImpl` objects and wraps
    them in a `Table` object. This is an inverse to the `unwrap_tables` function.
    """
    return traverse(
        arg, lambda x: table.Table(x) if isinstance(x, AbstractTableImpl) else x
    )
