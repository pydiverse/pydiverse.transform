from __future__ import annotations

from typing import NoReturn


def reraise(
    e: Exception,
    prefix: str | None = None,
    suffix: str | None = None,
) -> NoReturn:
    class ReraisedException(type(e)):
        def __init__(self, *args):
            Exception.__init__(self, *args)

        def __getattr__(self, item):
            return getattr(e, item)

        __repr__ = Exception.__repr__
        __str__ = Exception.__str__

    ReraisedException.__name__ = type(e).__name__
    ReraisedException.__qualname__ = type(e).__qualname__
    ReraisedException.__module__ = type(e).__module__

    suffix = "" if suffix is None else suffix
    prefix = "" if prefix is None else prefix

    if suffix != "":
        suffix = "\n" + suffix

    rre = ReraisedException(f"{prefix}{e}{suffix}")
    raise rre.with_traceback(e.__traceback__) from e.__cause__
