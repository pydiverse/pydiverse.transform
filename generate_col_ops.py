from __future__ import annotations

import os

from pydiverse.transform._internal.tree.registry import OperatorRegistry

path = "./src/pydiverse/transform/_internal/tree/col_expr.py"

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
                for op_name in OperatorRegistry.ALL_REGISTERED_OPS
            )
        ):
            in_generated_section = True
        elif in_col_expr_class and not (line.isspace() or line.startswith("    ")):
            for op in OperatorRegistry.ALL_REGISTERED_OPS:
                # TODO: use nice argument names here. An operator should have a
                # `arg_names` attr.
                new_file_contents += (
                    f"    def {op}(self,  *args, **kwargs):\n"
                    f'        return ColFn("{op}", self, *args, **kwargs)\n\n'
                )

            in_generated_section = False
            in_col_expr_class = False

        if not in_generated_section:
            new_file_contents += line

    file.seek(0)
    file.write(new_file_contents)
    file.truncate()

os.system(f"ruff format {path}")
