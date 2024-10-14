from __future__ import annotations

import os

from pydiverse.transform._internal.ops.operator import Operator
from pydiverse.transform._internal.ops.registry import OpRegistry
from pydiverse.transform._internal.ops.signature import Signature

path = "./src/pydiverse/transform/_internal/tree/col_expr.py"


def generate_fn_decl(name: str, op: Operator, sig: Signature) -> str:
    annotated_args = sum(
        f", {name}: ColExpr[{param.__name__}]"
        for param, name in zip(sig.params, op.arg_names, strict=True)
    )

    if op.context_kwargs is not None:
        annotated_kwargs = sum(
            f", {kwarg}: ColExpr | None = None" for kwarg in op.context_kwargs
        )
    else:
        annotated_kwargs = ""

    return f"    def {name}(self{annotated_args}{annotated_kwargs}):\n"


def generate_fn_body(op: Operator, sig: Signature):
    args = sum(f", {param.name}" for param in sig.params)

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
            and any(
                line[4:].startswith(f"def {op_name}")
                for op_name in OpRegistry.name_to_op.keys()
            )
        ):
            in_generated_section = True
        elif in_col_expr_class and not (line.isspace() or line.startswith("    ")):
            for op in OpRegistry.name_to_op.values():
                assert isinstance(op, Operator)

                if op.is_expression_method:
                    # TODO: use nice argument names here. An operator should have a
                    # `arg_names` attr.
                    new_file_contents += (
                        f"    def {op.name}(self,  *args, **kwargs):\n"
                        f'        return ColFn("{op.name}", self, *args, **kwargs)\n\n'
                    )

                    if op.has_rversion:
                        assert op.name.startswith("__") and op.name.endswith("__")
                        new_file_contents += (
                            f"    def __r{op.name[2:]}(self,  *args, **kwargs):\n"
                            f'        return ColFn("{op.name}", self, *args, **kwargs)'
                            "\n\n"
                        )

            in_generated_section = False
            in_col_expr_class = False

        if not in_generated_section:
            new_file_contents += line

    file.seek(0)
    file.write(new_file_contents)
    file.truncate()

os.system(f"ruff format {path}")
