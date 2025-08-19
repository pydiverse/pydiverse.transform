# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause

from pydiverse.transform._internal.util.warnings import warn

try:
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.sql import Join

    # 1) DB2-specific compilation for FULL OUTER JOIN
    @compiles(Join, "ibm_db_sa")
    def _compile_join_db2(join, compiler, **kwargs):
        # If this is a FULL join, force FULL OUTER JOIN keyword
        kwargs = kwargs.copy()
        kwargs["asfrom"] = True
        if getattr(join, "full", False):
            return "".join(
                (
                    compiler.process(join.left, **kwargs),
                    " FULL OUTER JOIN ",
                    compiler.process(join.right, **kwargs),
                    " ON ",
                    compiler.process(join.onclause, **kwargs),
                )
            )
        # Otherwise, fall back to default behavior for LEFT/INNER
        return compiler.visit_join(join, **kwargs)
except ImportError:
    warn("Failed to patch SQLAlchemy FULL OUTER JOIN for DB2. ")
