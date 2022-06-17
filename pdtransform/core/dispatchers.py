from .table import Table
from .table_impl import AbstractTableImpl
from functools import reduce, wraps, partial


class Pipeable:

    def __init__(self, f = None, calls = None):
        if f is not None:
            if calls is not None: raise ValueError
            self.calls = [f]
        else:
            self.calls = calls

    def __rshift__(self, other) -> 'Pipeable':
        """
        Pipeable >> other
        -> Lazy. Extend pipe.
        """
        if isinstance(other, Pipeable):
            return Pipeable(calls = self.calls + other.calls)
        elif callable(other):
            return Pipeable(calls = self.calls + [other])

        raise Exception

    def __rrshift__(self, other):
        """
        other >> Pipeable
        -> Eager.
        """
        if callable(other):
            return Pipeable(calls = [other] + self.calls)
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
    @wraps(func)
    def wrapper(*args, **kwargs):
        f = inverse_partial(func, *args, **kwargs)  # Apply arguments
        return Pipeable(f)
    return wrapper

def builtin_verb(backends = None):
    def wrap_and_unwrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return wrap_tables(func(*unwrap_tables(args), **unwrap_tables(kwargs)))
        return wrapper

    def check_backend(func):
        if backends is None:
            return func
        @wraps(func)
        def wrapper(*args, **kwargs):
            assert(len(args) > 0)
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
            f = check_backend(f)    # Check type of backend
            f = verb(f)             # Turn into pipable object
            return f(*args, **kwargs)
        return wrapper
    return decorator

# Helper

def unwrap_tables(arg=None):
    """
    Takes an instance or collection of `Table` objects and replaces them with
    their implementation.
    """
    if isinstance(arg, list):
        return [unwrap_tables(x) for x in arg]
    if isinstance(arg, tuple):
        return tuple(unwrap_tables(x) for x in arg)
    if isinstance(arg, dict):
        return { k: unwrap_tables(v) for k, v in arg.items() }
    return arg._impl if isinstance(arg, Table) else arg

def wrap_tables(arg=None):
    """
    Takes an instance or collection of `AbstractTableImpl` objects and wraps
    them in a `Table` object. This is an inverse to the `unwrap_tables` function.
    """
    if isinstance(arg, list):
        return [wrap_tables(x) for x in arg]
    if isinstance(arg, tuple):
        return tuple(wrap_tables(x) for x in arg)
    return Table(implementation = arg) if isinstance(arg, AbstractTableImpl) else arg