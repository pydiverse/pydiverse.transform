# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from functools import wraps

import pandas as pd
import polars as pl

from pydiverse.transform._internal import errors
from pydiverse.transform._internal.pipe.table import Table
from pydiverse.transform._internal.tree.col_expr import Col, ColExpr, EvalAligned


def aligned(fn=None, *, with_: str | None = None):
    """
    Decorator that automatically applies :doc:`pydiverse.transform.eval_aligned` to the
    return value of a function.

    :param with_:
        The table or column to align with.

    Examples
    --------
    >>> @aligned(with_="col")
    ... def reverse_col(col: pdt.Col) -> pdt.ColExpr:
    ...     return col.export(Polars).reverse()
    ...
    >>> t = pdt.Table(
    ...     {
    ...         "a": [1, 2, 3, 4],
    ...         "b": [2, 5, 16, 3],
    ...     },
    ...     name="t",
    ... )
    >>> t >> mutate(r=reverse_col(t.a)) >> show()
    Table `t` (backend: polars)
    shape: (4, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ r   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 2   ┆ 4   │
    │ 2   ┆ 5   ┆ 3   │
    │ 3   ┆ 16  ┆ 2   │
    │ 4   ┆ 3   ┆ 1   │
    └─────┴─────┴─────┘
    """

    errors.check_arg_type(str | None, "aligned", "with_", with_)

    def decorator(fn):
        signature = inspect.signature(fn)
        if with_ is not None and with_ not in signature.parameters:
            raise ValueError(f"function `{fn.__name__}` has no argument named `{with_}`")

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


def eval_aligned(val: ColExpr | pl.Series | pd.Series, with_: Table | Col | None = None) -> EvalAligned:
    """
    Allows to evaluate a column expression containing columns from different tables and
    to use polars / pandas Series in column expressions.

    :param val:
        The expression or polars / pandas Series to be aligned.

    :param with_:
        The table or column to align with.

    Examples
    --------
    Usage of a polars `Series` in a column expressions (the same works for pandas
    `Series`):

    >>> import polars as pl
    >>> t = pdt.Table(
    ...     {
    ...         "a": [1, 2, 3, 4],
    ...         "b": [2, 5, 16, 3],
    ...     },
    ...     name="t",
    ... )
    >>> s = pl.Series([9, 5, 4, 1])
    >>> t >> mutate(c=eval_aligned(t.a + s)) >> show()
    Table `t` (backend: polars)
    shape: (4, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 2   ┆ 10  │
    │ 2   ┆ 5   ┆ 7   │
    │ 3   ┆ 16  ┆ 7   │
    │ 4   ┆ 3   ┆ 5   │
    └─────┴─────┴─────┘

    Expression containing columns from different tables:

    >>> t1 = pdt.Table({"a": [1, 2, 3, 4]}, name="t1")
    >>> t2 = pdt.Table({"a": [5, 3, 1, 3]}, name="t2")
    >>> t1 >> mutate(c=eval_aligned(t1.a + t2.a, with_=t1)) >> show()
    Table `t1` (backend: polars)
    shape: (4, 2)
    ┌─────┬─────┐
    │ a   ┆ c   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 6   │
    │ 2   ┆ 5   │
    │ 3   ┆ 4   │
    │ 4   ┆ 7   │
    └─────┴─────┘
    """
    errors.check_arg_type(ColExpr | pl.Series | pd.Series, "eval_aligned", "val", val)
    errors.check_arg_type(Table | Col | None, "eval_aligned", "with_", with_)

    return EvalAligned(val, with_)
