from __future__ import annotations

import os
from collections.abc import Iterable

from pydiverse.transform._internal.backend.polars import PolarsImpl
from pydiverse.transform._internal.ops.core import NoExprMethod, Operator
from pydiverse.transform._internal.tree.dtypes import Dtype, Template
from pydiverse.transform._internal.tree.registry import Signature

path = "./src/pydiverse/transform/_internal/tree/col_expr.py"
reg = PolarsImpl.registry


def format_param(name: str, dtype: Dtype) -> str:
    if dtype.vararg:
        return f"*{name}"
    return name


def generate_fn_decl(op: Operator, sig: Signature, *, name=None) -> str:
    assert len(sig.params) >= 1
    if name is None:
        name = op.name

    defaults: Iterable = (
        op.defaults if op.defaults is not None else (... for _ in op.arg_names)
    )

    annotated_args = ", ".join(
        f"{format_param(name, param)}: "
        + (
            f"ColExpr[{param.__class__.__name__}]"
            if not isinstance(param, Template)
            else "ColExpr"
        )
        + (f" = {default_val}" if default_val is not ... else "")
        for param, name, default_val in zip(
            sig.params, op.arg_names, defaults, strict=True
        )
    )

    if op.context_kwargs is not None:
        # TODO: partition_by can only be col / col name, filter must be bool
        annotated_kwargs = "".join(
            f", {kwarg}: ColExpr | None = None" for kwarg in op.context_kwargs
        )
    else:
        annotated_kwargs = ""

    return f"    def {name}({annotated_args}{annotated_kwargs}):\n"


def generate_fn_body(op: Operator, sig: Signature):
    args = "".join(
        f", {format_param(name, param)}"
        for param, name in zip(sig.params, op.arg_names, strict=True)
    )

    if op.context_kwargs is not None:
        kwargs = "".join(f", {kwarg}" for kwarg in op.context_kwargs)
    else:
        kwargs = ""

    return f'        return ColFn("{op.name}", self{args}{kwargs})\n\n'


with open(path, "r+") as file:
    new_file_contents = ""
    in_col_expr_class = False
    in_generated_section = False

    for line in file:
        if line.startswith("class ColExpr"):
            in_col_expr_class = True
        elif (
            not in_generated_section
            and len(line) > 8
            and any(line[4:].startswith(f"def {op.name}") for op in reg.registered_ops)
        ):
            in_generated_section = True
        elif in_col_expr_class and not (line.isspace() or line.startswith("    ")):
            for op in reg.registered_ops:
                if isinstance(op, NoExprMethod):
                    continue
                new_file_contents += generate_fn_decl(
                    op, Signature.parse(op.signatures[0])
                ) + generate_fn_body(op, Signature.parse(op.signatures[0]))

                if op.name == "count":
                    # skip the nullary version of `count`
                    continue

                for sig in op.signatures[1:]:
                    new_file_contents += (
                        "    @overload\n"
                        + generate_fn_decl(op, Signature.parse(sig))
                        + "        ...\n\n"
                    )

            in_generated_section = False
            in_col_expr_class = False

        if not in_generated_section:
            new_file_contents += line

    file.seek(0)
    file.write(new_file_contents)
    file.truncate()

os.system(f"ruff format {path}")
