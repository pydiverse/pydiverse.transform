# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

import sqlalchemy as sqa
from sqlalchemy import Cast

from pydiverse.common import Decimal, Float
from pydiverse.transform._internal.backend.sql import SqlImpl
from pydiverse.transform._internal.ops import ops


class IbmDb2Impl(SqlImpl):
    backend_name = "ibm_db2"

    @classmethod
    def default_collation(cls) -> str | None:
        return None  # collation cannot be changed within expressions in DB2

    @classmethod
    def cast_compiled(cls, cast: Cast, compiled_expr: sqa.ColumnElement):
        _type = cls.sqa_type(cast.target_type)
        if isinstance(_type, sqa.String) and _type.length is None:
            # For DB2, we need to specify a length for string types.
            _type = sqa.String(length=32_672)
        # For SQLite, we ignore the `strict` parameter to `cast`.
        return sqa.cast(compiled_expr, _type)

    @classmethod
    def sqa_type(cls, pdt_type):
        if isinstance(pdt_type, Decimal):
            return sqa.DECIMAL(15, 6)
        return super().sqa_type(pdt_type)

    @classmethod
    def dialect_order_append_rand(cls):
        # DB2 hates non-deterministic behavior and forbids rand in ORDER BY clauses.
        return False


with IbmDb2Impl.impl_store.impl_manager as impl:

    @impl(ops.horizontal_min)
    def _horizontal_min(*x):
        if len(x) == 1:
            return sqa.func.LEAST(x[0], x[0])  # DB2 does not support LEAST with a single argument
        else:
            # the generated query will look extremely ugly but LEAST should be non-NULL
            # if any of the arguments is non-NULL
            any_non_null = sqa.func.COALESCE(*x)
            return sqa.func.LEAST(*[sqa.func.COALESCE(element, any_non_null) for element in x])

    @impl(ops.horizontal_max)
    def _horizontal_max(*x):
        if len(x) == 1:
            return sqa.func.GREATEST(x[0], x[0])  # DB2 does not support LEAST with a single argument
        else:
            # the generated query will look extremely ugly but LEAST should be non-NULL
            # if any of the arguments is non-NULL
            any_non_null = sqa.func.COALESCE(*x)
            return sqa.func.GREATEST(*[sqa.func.COALESCE(element, any_non_null) for element in x])

    @impl(ops.dt_second)
    def _dt_second(x):
        return sqa.func.cast(sqa.extract("second", x), type_=sqa.Integer())

    @impl(ops.dt_millisecond)
    def _dt_millisecond(x):
        return sqa.func.cast(
            (sqa.extract("second", x) * sqa.literal_column("1000.")),
            type_=sqa.Integer(),
        ) % sqa.literal_column("1000")

    @impl(ops.dt_microsecond)
    def _dt_microsecond(x):
        return sqa.func.cast(
            (sqa.extract("second", x) * sqa.literal_column("1000000.")),
            type_=sqa.Integer(),
        ) % sqa.literal_column("1000000")

    @impl(ops.dt_day_of_week)
    def _day_of_week(x):
        return (sqa.extract("dow", x) + 5) % sqa.literal_column("7") + 1

    @impl(ops.cbrt)
    def _cbrt(x):
        pow_impl = IbmDb2Impl.get_impl(ops.pow, (Float(), Float()))
        return sqa.func.sign(x) * pow_impl(sqa.func.abs(x), sqa.literal(1 / 3, type_=sqa.Double))

    @impl(ops.rand)
    def _rand():
        return sqa.func.RANDOM(1729)

    @impl(ops.str_contains)
    def _str_contains(x, pattern, allow_regex, true_if_regex_unsupported):
        if not allow_regex:
            return x.contains(pattern, autoescape=True)
        if pattern == "":
            return sqa.case(
                (x.is_(sqa.null()), sqa.null()),
                else_=sqa.literal(True, literal_execute=True),
            )
        return sqa.func.REGEXP_LIKE(x, pattern).cast(sqa.Boolean())
