from __future__ import annotations

import os

from pydiverse.transform._internal.backend.polars import PolarsImpl
from pydiverse.transform._internal.ops.core import Operator
from pydiverse.transform._internal.tree.dtypes import Dtype
from pydiverse.transform._internal.tree.registry import OperatorSignature

path = "./src/pydiverse/transform/_internal/tree/col_expr.py"
reg = PolarsImpl.registry


def format_param(name: str, dtype: Dtype) -> str:
    if dtype.vararg:
        return f"*{name}"
    return name


def generate_fn_decl(op: Operator, sig: OperatorSignature, *, name=None) -> str:
    if name is None:
        name = op.name

    annotated_args = sum(
        f", {format_param(name, param)}: ColExpr[{param.__name__}]"
        for param, name in zip(sig.params, op.arg_names, strict=True)
    )

    if op.context_kwargs is not None:
        annotated_kwargs = sum(
            f", {kwarg}: ColExpr | None = None" for kwarg in op.context_kwargs
        )
    else:
        annotated_kwargs = ""

    return f"    def {name}(self{annotated_args}{annotated_kwargs}):\n"


def generate_fn_body(op: Operator, sig: OperatorSignature):
    args = sum(
        f", {format_param(name, param)}"
        for param, name in zip(sig.params, op.arg_names, strict=True)
    )

    if op.context_kwargs is not None:
        kwargs = sum(f", {kwarg}" for kwarg in op.context_kwargs)
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
                new_file_contents += generate_fn_decl(
                    op, OperatorSignature.parse(op.signatures[0])
                ) + generate_fn_body(op, OperatorSignature.parse(op.signatures[0]))

            in_generated_section = False
            in_col_expr_class = False

        if not in_generated_section:
            new_file_contents += line

    file.seek(0)
    file.write(new_file_contents)
    file.truncate()

os.system(f"ruff format {path}")
