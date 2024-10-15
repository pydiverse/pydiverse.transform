from __future__ import annotations

import os
from collections.abc import Iterable
from types import NoneType

from pydiverse.transform._internal.backend.polars import PolarsImpl
from pydiverse.transform._internal.ops.core import NoExprMethod, Operator
from pydiverse.transform._internal.tree.dtypes import (
    Dtype,
    Template,
    pdt_type_to_python,
)
from pydiverse.transform._internal.tree.registry import Signature

col_expr_path = "./src/pydiverse/transform/_internal/tree/col_expr.py"
fns_path = "./src/pydiverse/transform/_internal/pipe/functions.py"
reg = PolarsImpl.registry
namespaces = ["str", "dt"]
rversions = {
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__floordiv__",
    "__pow__",
    "__mod__",
    "__and__",
    "__or__",
    "__xor__",
}


def format_param(name: str, dtype: Dtype) -> str:
    if dtype.vararg:
        return f"*{name}"
    return name


def type_annotation(param: Dtype, specialize_generic: bool) -> str:
    if not specialize_generic or isinstance(param, Template):
        return "ColExpr"
    if param.const:
        python_type = pdt_type_to_python(param)
        return python_type.__name__ if python_type is not NoneType else "None"
    return f"ColExpr[{param.__class__.__name__}]"


def generate_fn_decl(
    op: Operator, sig: Signature, *, name=None, specialize_generic: bool = True
) -> str:
    if name is None:
        name = op.name

    defaults: Iterable = (
        op.defaults if op.defaults is not None else (... for _ in op.arg_names)
    )

    annotated_args = ", ".join(
        f"{format_param(name, param)}: "
        + type_annotation(param, specialize_generic)
        + (f" = {default_val}" if default_val is not ... else "")
        for param, name, default_val in zip(
            sig.params, op.arg_names, defaults, strict=True
        )
    )

    if op.context_kwargs is not None:
        context_kwarg_annotation = {
            "partition_by": "Col | ColName | Iterable[Col | ColName]",
            "arrange": "ColExpr | Iterable[ColExpr]",
            "filter": "ColExpr[Bool] | Iterable[ColExpr[Bool]]",
        }

        annotated_kwargs = "".join(
            f", {kwarg}: {context_kwarg_annotation[kwarg]} | None = None"
            for kwarg in op.context_kwargs
        )

        if len(sig.params) == 0 or not sig.params[-1].vararg:
            annotated_kwargs = "*" + annotated_kwargs
            if len(sig.params) > 0:
                annotated_kwargs = ", " + annotated_kwargs
    else:
        annotated_kwargs = ""

    return (
        f"def {name}({annotated_args}{annotated_kwargs}) "
        f"-> {type_annotation(sig.return_type, specialize_generic)}:\n"
    )


def generate_fn_body(
    op: Operator,
    sig: Signature,
    arg_names: list[str] | None = None,
    *,
    rversion: bool = False,
):
    if arg_names is None:
        arg_names = op.arg_names

    if rversion:
        assert len(arg_names) == 2
        assert not any(param.vararg for param in sig.params)
        arg_names = list(reversed(arg_names))

    args = "".join(
        f", {format_param(name, param)}"
        for param, name in zip(sig.params, arg_names, strict=True)
    )

    if op.context_kwargs is not None:
        kwargs = "".join(f", {kwarg}={kwarg}" for kwarg in op.context_kwargs)
    else:
        kwargs = ""

    return f'    return ColFn("{op.name}"{args}{kwargs})\n\n'


def generate_overloads(
    op: Operator, *, name: str | None = None, rversion: bool = False
):
    res = ""
    in_namespace = "." in op.name
    if name is None:
        name = op.name if not in_namespace else op.name.split(".")[1]

    has_overloads = len(op.signatures) > 1
    if has_overloads:
        for sig in op.signatures:
            res += (
                "@overload\n"
                + generate_fn_decl(op, Signature.parse(sig), name=name)
                + "    ...\n\n"
            )

    res += generate_fn_decl(
        op,
        Signature.parse(op.signatures[0]),
        name=name,
        specialize_generic=not has_overloads,
    ) + generate_fn_body(
        op,
        Signature.parse(op.signatures[0]),
        ["self.arg"] + op.arg_names[1:] if in_namespace else None,
        rversion=rversion,
    )

    return res


def indent(s: str, by: int) -> str:
    return "".join(" " * by + line + "\n" for line in s.split("\n"))


with open(col_expr_path, "r+") as file:
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
            for op_name in sorted(PolarsImpl.registry.ALL_REGISTERED_OPS):
                op = PolarsImpl.registry.get_op(op_name)
                if isinstance(op, NoExprMethod):
                    continue

                op_overloads = generate_overloads(op)
                if op_name in rversions:
                    op_overloads += generate_overloads(
                        op, name=f"__r{op_name[2:]}", rversion=True
                    )

                op_overloads = indent(op_overloads, 4)

                if "." in op.name:
                    namespace_contents[op.name.split(".")[0]] += op_overloads
                else:
                    new_file_contents += op_overloads

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

os.system(f"ruff format {col_expr_path}")

with open(fns_path, "r+") as file:
    new_file_contents = ""
    display_name = {"hmin": "min", "hmax": "max"}

    for line in file:
        new_file_contents += line
        if line.startswith("    return LiteralCol"):
            for op_name in sorted(PolarsImpl.registry.ALL_REGISTERED_OPS):
                op = PolarsImpl.registry.get_op(op_name)
                if not isinstance(op, NoExprMethod):
                    continue

                new_file_contents += generate_overloads(
                    op, name=display_name.get(op_name)
                )
            break

    file.seek(0)
    file.write(new_file_contents)
    file.truncate()

os.system(f"ruff format {fns_path}")
