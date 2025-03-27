from __future__ import annotations

import os
from collections.abc import Iterable
from types import NoneType

from pydiverse.common import Dtype
from pydiverse.transform._internal.ops import ops
from pydiverse.transform._internal.ops.op import Operator
from pydiverse.transform._internal.ops.signature import Signature
from pydiverse.transform._internal.tree import types
from pydiverse.transform._internal.tree.types import Tyvar

COL_EXPR_PATH = "./src/pydiverse/transform/_internal/tree/col_expr.py"
FNS_PATH = "./src/pydiverse/transform/_internal/pipe/functions.py"
API_DOCS_PATH = "./docs/source/reference/operators/index.rst"

NAMESPACES = ["str", "dt", "dur", "list"]

RVERSIONS = {
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


def add_vararg_star(formatted_args: str) -> str:
    last_arg = "*" + formatted_args.split(", ")[-1]
    return ", ".join(formatted_args.split(", ")[:-1] + [last_arg])


def type_annotation(dtype: Dtype, specialize_generic: bool) -> str:
    if (not specialize_generic and not types.is_const(dtype)) or isinstance(
        types.without_const(dtype), Tyvar
    ):
        return "ColExpr"
    if types.is_const(dtype):
        python_type = types.to_python(dtype)
        return python_type.__name__ if python_type is not NoneType else "None"
    return f"ColExpr[{dtype.__class__.__name__}]"


def generate_fn_decl(
    op: Operator, sig: Signature, *, name=None, specialize_generic: bool = True
) -> str:
    if name is None:
        name = op.name

    defaults: Iterable = (
        op.default_values
        if op.default_values is not None
        else (... for _ in op.param_names)
    )

    annotated_args = ", ".join(
        name
        + ": "
        + type_annotation(dtype, specialize_generic)
        + (f" = {repr(default_val)}" if default_val is not ... else "")
        for dtype, name, default_val in zip(
            sig.types, op.param_names, defaults, strict=True
        )
    )
    if sig.is_vararg:
        annotated_args = add_vararg_star(annotated_args)

    if len(op.context_kwargs) > 0:
        context_kwarg_annotation = {
            "partition_by": "Col | ColName | str | Iterable[Col | ColName | str]",
            "arrange": "ColExpr | Iterable[ColExpr]",
            "filter": "ColExpr[Bool] | Iterable[ColExpr[Bool]]",
        }

        annotated_kwargs = "".join(
            f", {kwarg.name}: {context_kwarg_annotation[kwarg.name]}"
            + f"{'' if kwarg.required else ' | None = None'}"
            for kwarg in op.context_kwargs
        )

        if len(sig.types) == 0 or not sig.is_vararg:
            annotated_kwargs = "*" + annotated_kwargs
            if len(sig.types) > 0:
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
    param_names: list[str] | None = None,
    *,
    op_var_name: str,
    rversion: bool = False,
):
    if param_names is None:
        param_names = op.param_names

    if rversion:
        assert len(param_names) == 2
        assert not sig.is_vararg
        param_names = list(reversed(param_names))

    args = "".join(f", {name}" for name in param_names)
    if sig.is_vararg:
        args = add_vararg_star(args)

    if op.context_kwargs is not None:
        kwargs = "".join(f", {kwarg.name}={kwarg.name}" for kwarg in op.context_kwargs)
    else:
        kwargs = ""

    return f"    return ColFn(ops.{op_var_name}{args}{kwargs})\n\n"


def generate_overloads(
    op: Operator, *, name: str | None = None, rversion: bool = False, op_var_name: str
):
    res = ""
    in_namespace = "." in op.name
    if name is None:
        name = op.name if not in_namespace else op.name.split(".")[1]

    has_overloads = len(op.signatures) > 1
    if has_overloads:
        for sig in op.signatures:
            res += "@overload\n" + generate_fn_decl(op, sig, name=name) + "    ...\n\n"

    res += (
        generate_fn_decl(
            op, op.signatures[0], name=name, specialize_generic=not has_overloads
        )
        + f'    """{op.doc}"""\n\n'
        + generate_fn_body(
            op,
            op.signatures[0],
            ["self.arg"] + op.param_names[1:] if in_namespace else None,
            rversion=rversion,
            op_var_name=op_var_name,
        )
    )

    return res


def indent(s: str, by: int) -> str:
    return "".join(" " * by + line + "\n" for line in s.split("\n"))


with open(COL_EXPR_PATH, "r+") as file:
    new_file_contents = ""
    in_generated_section = False
    namespace_contents: dict[str, str] = {
        name: (
            f'@register_accessor("{name}")\n'
            "@dataclasses.dataclass(slots=True)\n"
            f"class {name.title()}Namespace(FnNamespace):\n"
        )
        for name in NAMESPACES
    }

    for line in file:
        if not in_generated_section and line.startswith(
            "    # --- generated code starts here, do not delete this comment ---"
        ):
            in_generated_section = True
            new_file_contents += (
                "    # --- generated code starts here, do not delete this "
                "comment ---\n\n"
            )
        elif in_generated_section and line.startswith(
            "# --- generated code ends here, do not delete this comment ---"
        ):
            for op_var_name in sorted(ops.__dict__):
                op = ops.__dict__[op_var_name]
                if not isinstance(op, Operator) or not op.generate_expr_method:
                    continue

                op_overloads = generate_overloads(op, op_var_name=op_var_name)
                if op.name in RVERSIONS:
                    op_overloads += generate_overloads(
                        op,
                        name=f"__r{op.name[2:]}",
                        rversion=True,
                        op_var_name=op_var_name,
                    )

                op_overloads = indent(op_overloads, 4)

                if "." in op.name:
                    namespace_contents[op.name.split(".")[0]] += op_overloads
                else:
                    new_file_contents += op_overloads

            for name in NAMESPACES:
                new_file_contents += f"    {name} : {name.title()}Namespace\n"

            new_file_contents += (
                "@dataclasses.dataclass(slots=True)\n"
                "class FnNamespace:\n"
                "    arg: ColExpr\n"
            )

            for name in NAMESPACES:
                new_file_contents += namespace_contents[name]

            in_generated_section = False

        if not in_generated_section:
            new_file_contents += line

    file.seek(0)
    file.write(new_file_contents)
    file.truncate()

os.system(f"ruff format {COL_EXPR_PATH}")


with open(FNS_PATH, "r+") as file:
    new_file_contents = ""

    for line in file:
        new_file_contents += line
        if line.startswith(
            "# --- from here the code is generated, do not delete this comment ---"
        ):
            new_file_contents += "\n"
            for op_var_name in sorted(ops.__dict__):
                op = ops.__dict__[op_var_name]
                if isinstance(op, Operator) and not op.generate_expr_method:
                    new_file_contents += generate_overloads(op, op_var_name=op_var_name)
            break

    file.seek(0)
    file.write(new_file_contents)
    file.truncate()

os.system(f"ruff format {FNS_PATH}")

with open(API_DOCS_PATH, "r+") as file:
    new_file_contents = ""

    for line in file:
        new_file_contents += line
        if line.startswith("Expression methods"):
            new_file_contents += (
                "------------------\n\n"
                ".. currentmodule:: pydiverse.transform.ColExpr\n\n"
                ".. autosummary::\n"
                "   :nosignatures:\n\n   "
            )

            new_file_contents += "\n   ".join(
                sorted(
                    [
                        op.name
                        for op in ops.__dict__.values()
                        if isinstance(op, Operator) and op.generate_expr_method
                    ]
                    + ["rank", "dense_rank", "map", "cast"]
                )
            )

            new_file_contents += (
                "\n\nGlobal functions\n"
                "----------------\n\n"
                ".. currentmodule:: pydiverse.transform\n\n"
                ".. autosummary::\n"
                "   :nosignatures:\n\n   "
            )

            new_file_contents += (
                "\n   ".join(
                    sorted(
                        [
                            op.name
                            for op in ops.__dict__.values()
                            if isinstance(op, Operator) and not op.generate_expr_method
                        ]
                        + ["when", "lit"]
                    )
                )
                + "\n"
            )

            break

    file.seek(0)
    file.write(new_file_contents)
    file.truncate()
