# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from functools import wraps

import polars as pl

from pydiverse.transform._internal import errors
from pydiverse.transform._internal.pipe.table import Table
from pydiverse.transform._internal.tree.col_expr import ColExpr, EvalAligned


def aligned(fn=None, *, with_: str | None = None):
    errors.check_arg_type(str | None, "aligned", "with_", with_)

    def decorator(fn):
        signature = inspect.signature(fn)
        if with_ is not None and with_ not in signature.parameters:
            raise ValueError(
                f"function `{fn.__name__}` has no argument named `{with_}`"
            )

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if with_ is not None:
                bound_sig = signature.bind(*args, **kwargs)
                bound_sig.apply_defaults()
                with_obj = bound_sig.arguments[with_]
            else:
                with_obj = None

            return eval_aligned(fn(*args, **kwargs), with_=with_obj)

        return wrapper

    if fn is not None:
        return decorator(fn)

    return decorator


# TODO: documentation + allow cols for with_
def eval_aligned(data: ColExpr | pl.Series, with_: Table | None = None) -> EvalAligned:
    errors.check_arg_type(ColExpr | pl.Series, "eval_aligned", "data", data)
    errors.check_arg_type(Table | None, "eval_aligned", "with_", with_)

    return EvalAligned(data, with_)
