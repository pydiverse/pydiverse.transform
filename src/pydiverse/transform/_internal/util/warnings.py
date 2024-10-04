from __future__ import annotations

import inspect
import sys
import warnings as py_warnings

from pydiverse.transform._internal.errors import NonStandardWarning


def warn(
    message: str,
    category: type[Warning] = None,
    stacklevel=1,
):
    py_warnings.warn(message, category, stacklevel=stacklevel + 1)
    return

    stack = inspect.stack(context=0)
    frame = stack[stacklevel]

    # for f in stack[stacklevel:]:
    #     if frame_self := f.frame.f_locals.get("self"):
    #         if isinstance(frame_self, AbstractTableImpl.ExpressionCompiler):
    #             table_impl = frame_self.backend
    #             break

    frame_globals = frame.frame.f_globals
    filename = frame.filename
    lineno = frame.lineno

    del frame  # Prevent reference cycle
    del stack  # Prevent reference cycle

    registry = frame_globals.setdefault("__pydiverse_transform_warnings_registry__", {})
    key = (message, category, lineno)

    if registry.get(key):
        # Not a new warning
        return
    registry[key] = 1

    print(f"{filename}:{lineno}: {category.__name__}: {message}", file=sys.stderr)


def warn_non_standard(
    message: str,
    stacklevel=1,
):
    warn(
        message,
        category=NonStandardWarning,
        stacklevel=stacklevel + 1,
    )
