from __future__ import annotations

import copy
from functools import partial, reduce, wraps
from typing import Any

from pydiverse.transform.core.expressions import (
    Col,
    ColName,
)
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

        raise RuntimeError

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
    from pydiverse.transform.core.table import Table

    def copy_tables(arg: Any = None):
        return traverse(arg, lambda x: copy.copy(x) if isinstance(x, Table) else x)

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
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            f = func
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
    from pydiverse.transform.core.verbs import select

    if isinstance(arg, Col):
        table = (arg.table >> select(arg))._impl
        col = table.get_col(arg)

        table.available_cols = {col.uuid}
        table.named_cols = bidict({col.name: col.uuid})
        return table
    elif isinstance(arg, ColName):
        raise ValueError("Can't start a pipe with a lambda column.")

    return arg


def unwrap_tables(arg: Any = None):
    """
    Takes an instance or collection of `Table` objects and replaces them with
    their implementation.
    """
    from pydiverse.transform.core.table import Table

    return traverse(arg, lambda x: x._impl if isinstance(x, Table) else x)


def wrap_tables(arg: Any = None):
    """
    Takes an instance or collection of `AbstractTableImpl` objects and wraps
    them in a `Table` object. This is an inverse to the `unwrap_tables` function.
    """
    from pydiverse.transform.core.table import Table
    from pydiverse.transform.core.table_impl import AbstractTableImpl

    return traverse(arg, lambda x: Table(x) if isinstance(x, AbstractTableImpl) else x)
