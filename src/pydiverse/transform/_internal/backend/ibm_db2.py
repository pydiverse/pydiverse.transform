# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
import sqlalchemy as sqa
from sqlalchemy import Cast

from pydiverse.transform._internal.backend.sql import SqlImpl


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
