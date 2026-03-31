# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import copy
from functools import partial, wraps

from pydiverse.transform._internal.errors import SubqueryError
from pydiverse.transform._internal.tree import verbs
from pydiverse.transform._internal.tree.col_expr import Col


class Pipeable:
    def __init__(self, f=None, calls=None):
        if f is not None:
            if calls is not None:
                raise ValueError
            self.calls = [f]
        else:
            self.calls = calls

    def __rshift__(self, other) -> "Pipeable":
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
    """
    Decorator for creating verbs.

    A verb is simply a function decorated with this decorator that takes a
    pydiverse.transform Table as the first argument. `@verb` enables usage of
    the function with the pipe `>>` syntax.

    Examples
    --------
    >>> @verb
    ... def strip_all_strings(tbl: pdt.Table) -> pdt.Table:
    ...     return tbl >> mutate(
    ...         **{c.name: c.str.strip() for c in tbl if c.dtype() == pdt.String()}
    ...     )
    >>> t = pdt.Table(
    ...     {
    ...         "a": ["  abcd 5 ", "212"],
    ...         "b": [" 917. __ ", " 2 "],
    ...         "c": [1, 2],
    ...     },
    ...     name="t",
    ... )
    >>> t >> strip_all_strings() >> show()
    Table `t` (backend: polars)
    shape: (2, 3)
    ┌─────┬────────┬─────────┐
    │ c   ┆ a      ┆ b       │
    │ --- ┆ ---    ┆ ---     │
    │ i64 ┆ str    ┆ str     │
    ╞═════╪════════╪═════════╡
    │ 1   ┆ abcd 5 ┆ 917. __ │
    │ 2   ┆ 212    ┆ 2       │
    └─────┴────────┴─────────┘

    Note
    ----
    To make the code completion of your IDE not show the table as the first argument,
    simply add an `@overload` before the verb definition:
    >>> @overload
    ... def strip_all_strings() -> pdt.Table: ...
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return Pipeable(inverse_partial(fn, *args, **kwargs))

    return wrapper


def check_subquery(new_tbl, child_tbl, *, is_right: bool = False):
    reason = child_tbl._cache.requires_subquery(new_tbl._ast)
    if reason is not None:
        # Search among descendants of the current node for an `Alias` that can be used
        # to create a subquery. If we hit a `Join` or `SubqueryMarker`, we stop.
        # First check if the child itself is an Alias (common case)
        if isinstance(child_tbl._ast, verbs.Alias):
            nd = child_tbl._ast
            # Create a new Join with SubqueryMarker as child
            subquery_marker = verbs.SubqueryMarker(nd)
            if is_right:
                assert isinstance(new_tbl._ast, verbs.Join)
                new_join = verbs.Join(
                    new_tbl._ast.child,
                    subquery_marker,
                    new_tbl._ast.on,
                    new_tbl._ast.how,
                    new_tbl._ast.validate,
                )
                new_join.name = new_tbl._ast.name
                subquery_node = new_join.right
            else:
                assert isinstance(new_tbl._ast, verbs.Join)
                new_join = verbs.Join(
                    subquery_marker,
                    new_tbl._ast.right,
                    new_tbl._ast.on,
                    new_tbl._ast.how,
                    new_tbl._ast.validate,
                )
                new_join.name = new_tbl._ast.name
                subquery_node = new_join.child

            from pydiverse.transform._internal.pipe.table import Table

            test_tbl = Table(subquery_node)
            new_join.map_col_nodes(
                lambda expr: test_tbl._cache.cols[expr._uuid]  # noqa: B023
                if isinstance(expr, Col) and expr._uuid in test_tbl._cache.cols  # noqa: B023
                else expr
            )
            modified_new_tbl = copy.copy(new_tbl)
            modified_new_tbl._ast = new_join
            return (modified_new_tbl, test_tbl)

        chain: list[verbs.Verb] = [new_tbl._ast]
        for nd in child_tbl._ast.iter_subtree_preorder():
            if isinstance(nd, verbs.Alias):
                # Create a new Join with SubqueryMarker as child
                # Don't clone the Join - create a new one with the SubqueryMarker
                subquery_marker = verbs.SubqueryMarker(nd)
                if is_right:
                    assert isinstance(new_tbl._ast, verbs.Join)
                    # Create a new Join with the same attributes but SubqueryMarker as right child
                    new_join = verbs.Join(
                        new_tbl._ast.child,  # left child stays the same
                        subquery_marker,  # right child is SubqueryMarker
                        new_tbl._ast.on,
                        new_tbl._ast.how,
                        new_tbl._ast.validate,
                    )
                    new_join.name = new_tbl._ast.name
                    subquery_node = new_join.right
                else:
                    # Create a new Join with SubqueryMarker as left child
                    assert isinstance(new_tbl._ast, verbs.Join)
                    new_join = verbs.Join(
                        subquery_marker,  # left child is SubqueryMarker
                        new_tbl._ast.right,  # right child stays the same
                        new_tbl._ast.on,
                        new_tbl._ast.how,
                        new_tbl._ast.validate,
                    )
                    new_join.name = new_tbl._ast.name
                    subquery_node = new_join.child

                from pydiverse.transform._internal.pipe.table import Table

                # Create a test table from the SubqueryMarker to get its cache (which should have partition_by cleared)
                test_tbl = Table(subquery_node)

                # Update column references in the new AST to point to columns from the subquery
                new_join.map_col_nodes(
                    lambda expr: test_tbl._cache.cols[expr._uuid]  # noqa: B023
                    if isinstance(expr, Col) and expr._uuid in test_tbl._cache.cols  # noqa: B023
                    else expr
                )

                # SubqueryMarker should have cleared partition_by and group_by
                # Use the modified AST with the SubqueryMarker
                modified_new_tbl = copy.copy(new_tbl)
                modified_new_tbl._ast = new_join
                return (modified_new_tbl, test_tbl)

            if isinstance(nd, verbs.SubqueryMarker | verbs.Join):
                break
            chain.append(nd)

        raise SubqueryError(
            f"Executing the `{new_tbl._ast.__class__.__name__.lower()}` verb "
            f"on the table `{new_tbl._ast.child.name}` requires a subquery, which "
            "is forbidden in transform by default.\n"
            f"reason for the subquery: {reason}\n"
            f"hint: Materialize the table `{new_tbl._ast.child.name}` before this "
            "verb. If you are sure you want to do a subquery, put an "
            "`>> alias()` before this verb. "
        )

    return (new_tbl, child_tbl)


# Checks for subqueries and updates the cache for all verbs except `join`. Since `join`
# is much more complex, we do these tasks manually in the verb.
def modify_ast(fn):
    @wraps(fn)
    def _fn(table, *args, **kwargs):
        new = fn(table, *args, **kwargs)
        assert new._ast != table._ast

        assert new._ast.child == table._ast
        new, child = check_subquery(new, table)
        new._cache = child._cache.update(new._ast)

        return new

    return _fn
