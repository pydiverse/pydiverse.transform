from __future__ import annotations

from functools import partial, reduce, wraps


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


# TODO: validate that the first arg is a table here


def verb(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return Pipeable(inverse_partial(fn, *args, **kwargs))

    return wrapper


def builtin_verb(backends=None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return Pipeable(inverse_partial(fn, *args, **kwargs))

        return wrapper

    return decorator
