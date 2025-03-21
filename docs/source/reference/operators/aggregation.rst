===========
Aggregation
===========

Aggregation functions take a ``partition_by`` and ``filter`` keyword argument. The
``partition_by`` argument can only be given when used within ``mutate``. If a
``partition_by`` argument is given and there is a surrounding ``group_by`` /
``ungroup``, the ``group_by`` is ignored and the value of ``partition_by`` is used.

.. warning::
    The ``filter`` argument works similar to ``Expr.filter`` in polars. But in contrast
    to polars, if all values in a group are ``null`` or the group becomes empty after
    filtering, the value of every aggregation function for that group is ``null``, too.

.. currentmodule:: pydiverse.transform.ColExpr
.. autosummary::
    :toctree: _generated/
    :nosignatures:
    :template: short_title.rst

    count
    all
    any
    max
    mean
    min
    sum
    str.join
