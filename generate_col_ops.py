from __future__ import annotations

import os
from collections.abc import Iterable

from pydiverse.transform._internal.backend.polars import PolarsImpl
from pydiverse.transform._internal.ops.core import NoExprMethod, Operator
from pydiverse.transform._internal.tree.dtypes import Dtype, Template
from pydiverse.transform._internal.tree.registry import Signature

path = "./src/pydiverse/transform/_internal/tree/col_expr.py"
reg = PolarsImpl.registry
namespaces = ["str", "dt"]


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

        if not sig.params[-1].vararg:
            annotated_kwargs = ", *" + annotated_kwargs
    else:
        annotated_kwargs = ""

    return f"    def {name}({annotated_args}{annotated_kwargs}):\n"


def generate_fn_body(op: Operator, sig: Signature, arg_names: list[str] | None = None):
    if arg_names is None:
        arg_names = op.arg_names

    args = "".join(
        f", {format_param(name, param)}"
        for param, name in zip(sig.params, arg_names, strict=True)
    )

    if op.context_kwargs is not None:
        kwargs = "".join(f", {kwarg}={kwarg}" for kwarg in op.context_kwargs)
    else:
        kwargs = ""

    return f'        return ColFn("{op.name}"{args}{kwargs})\n\n'


with open(path, "r+") as file:
    new_file_contents = ""
    in_col_expr_class = False
    in_generated_section = False
    namespace_contents: dict[str, str] = {
        name: (
            "@dataclasses.dataclass(slots=True)\n"
            f"class {name.title()}Namespace(FnNamespace):\n"
        )
        for name in namespaces
    }

    for line in file:
        if line.startswith("class ColExpr"):
            in_col_expr_class = True
        elif not in_generated_section and line.startswith("    @overload"):
            in_generated_section = True
        elif in_col_expr_class and line.startswith("class Col"):
            for op in sorted(reg.registered_ops, key=lambda op: op.name):
                if isinstance(op, NoExprMethod):
                    continue

                op_definition = ""
                in_namespace = "." in op.name
                method_name = op.name if not in_namespace else op.name.split(".")[1]

                if op.name != "count":
                    for sig in op.signatures[1:]:
                        op_definition += (
                            "    @overload\n"
                            + generate_fn_decl(
                                op, Signature.parse(sig), name=method_name
                            )
                            + "        ...\n\n"
                        )

                op_definition += generate_fn_decl(
                    op, Signature.parse(op.signatures[0]), name=method_name
                ) + generate_fn_body(
                    op,
                    Signature.parse(op.signatures[0]),
                    ["self.arg"] + op.arg_names[1:] if in_namespace else None,
                )

                if in_namespace:
                    namespace_contents[op.name.split(".")[0]] += op_definition
                else:
                    new_file_contents += op_definition

            for name in namespaces:
                new_file_contents += (
                    "    @property\n"
                    f"    def {name}(self):\n"
                    f"        return {name.title()}Namespace(self)\n\n"
                )

            new_file_contents += (
                "@dataclasses.dataclass(slots=True)\n"
                "class FnNamespace:\n"
                "    arg: ColExpr\n"
            )

            for name in namespaces:
                new_file_contents += namespace_contents[name]

            in_generated_section = False
            in_col_expr_class = False

        if not in_generated_section:
            new_file_contents += line

    file.seek(0)
    file.write(new_file_contents)
    file.truncate()

os.system(f"ruff format {path}")
