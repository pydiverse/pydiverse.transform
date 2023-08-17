from __future__ import annotations

import copy
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from random import Random

from pydiverse.transform import Table, ops, λ
from pydiverse.transform.core import dtypes
from pydiverse.transform.core import expressions as exp
from pydiverse.transform.core.registry import OperatorSignature
from pydiverse.transform.ops import OPType


def _get_table_schema(table: Table):
    return {col._.name: col._.dtype for col in table}


def _get_available_ops(table: Table):
    op_reg = table._impl.operator_registry
    op_signatures = {}

    for op_name in sorted(op_reg.ALL_REGISTERED_OPS):
        try:
            op = op_reg.get_operator(op_name)
            op_signatures[op] = [OperatorSignature.parse(sig) for sig in op.signatures]
        except Exception:
            warnings.warn(f"Operator not implemented {op_name}")
            continue

    return op_signatures


def _expand_templates_in_signature(
    signature: OperatorSignature, assignment: dict
) -> OperatorSignature:
    s = copy.deepcopy(signature)

    for i, arg in enumerate(signature.args):
        if isinstance(arg, dtypes.Template):
            s.args[i] = copy.copy(assignment[arg.name])
            s.args[i].const = arg.const
            s.args[i].vararg = arg.vararg

    if isinstance(signature.rtype, dtypes.Template):
        s.rtype = copy.copy(assignment[signature.rtype.name])

    return s


def _get_all_signature_template_expansions(
    signature: OperatorSignature, available_dtypes: list[dtypes.DType]
) -> Iterable[OperatorSignature]:
    templates = set()
    for arg in signature.args:
        if isinstance(arg, dtypes.Template):
            templates.add(arg.name)

    if len(templates) == 0:
        yield signature
        return

    templates = sorted(list(templates))
    for combination in itertools.combinations_with_replacement(
        available_dtypes, len(templates)
    ):
        assignment = dict(zip(templates, combination))  # type: ignore
        yield _expand_templates_in_signature(signature, assignment)


class BaseExpressionFuzzer:
    def __init__(
        self,
        table: Table,
        seed: int = 0xBADC0DE,
    ):
        self.random = Random(seed)
        self.table_schema = _get_table_schema(table)

        # Expression Leaves

        self.cols: dict[dtypes.DType, list] = defaultdict(lambda: [])
        for name, dtype in self.table_schema.items():
            self.cols[dtype.without_modifiers()].append(λ[name])

        self.literals: dict[dtypes.DType, list] = {
            dtypes.Int(): [-100, -7, -3, -1, 0, 1, 3, 7, 100],
            dtypes.Float(): [-1.5e10, -1.5, -1.5e-5, 0.0, 1.5e-5, 1.5e10],
            dtypes.String(): ["", " ", "a", "A", "foobar"],
            dtypes.Bool(): [False, True],
            dtypes.DateTime(): [
                datetime(2023, 7, 31, 10, 16, 13),
                datetime(1970, 1, 1),
                datetime(2200, 1, 1),
            ],
        }

        self.available_dtypes = sorted(self.literals.keys(), key=lambda e: e.name)

        # Ops

        self.available_ops = {
            op: sigs
            for op, sigs in _get_available_ops(table).items()
            if self._filter_op(op, sigs)
        }

        self.ops_by_rtype: dict[
            dtypes.DType, list[tuple[ops.Operator, OperatorSignature]]
        ] = defaultdict(lambda: [])

        for op, sigs in self.available_ops.items():
            for sig in sigs:
                for expanded_sig in _get_all_signature_template_expansions(
                    sig, self.available_dtypes
                ):
                    self.ops_by_rtype[expanded_sig.rtype].append((op, expanded_sig))

    # Generation

    def generate_expression(self, *, max_depth: int = 10, **kwargs):
        if "dtype" not in kwargs:
            kwargs["dtype"] = self._dtype()
        if "depth" not in kwargs:
            kwargs["depth"] = self.random.randint(1, max_depth)
        return self._expression(**kwargs)

    def generate_expressions(self, **kwargs):
        while True:
            yield self.generate_expression(**kwargs)

    def _expression(
        self,
        dtype: dtypes.DType,
        depth: int,
        can_be_literal=True,
        can_be_window=True,
        can_be_aggregate=True,
    ):
        assert depth >= 1

        if depth == 1:
            if not can_be_literal:
                return self._column(dtype)

            if self.random.random() <= 0.1:
                return exp.SymbolicExpression(self._literal(dtype))
            return self._column(dtype)

        # Combine expressions
        available_ops = [
            (op, sig)
            for op, sig in self.ops_by_rtype[dtype.without_modifiers()]
            if (op.ftype != OPType.WINDOW or can_be_window)
            and (op.ftype != OPType.AGGREGATE or can_be_aggregate)
        ]

        if len(available_ops) == 0:
            return self._expression(dtype, 1, can_be_literal=can_be_literal)

        op, sig = self.random.choice(available_ops)
        if special_case := self._special_case(op, sig, dtype, depth):
            return exp.SymbolicExpression(special_case)

        # Update can_be_window, can_be_aggregate for children
        can_be_window &= op.ftype == OPType.EWISE
        can_be_aggregate &= op.ftype == OPType.EWISE

        # At least one of the arguments must be of depth `depth-1`
        arg_depth = []
        if len(sig.args) > 0:
            arg_depth = [self.random.randint(1, depth - 1) for _ in sig.args]
            arg_depth[self.random.randint(0, len(sig.args) - 1)] = depth - 1

        arguments = []
        for arg_i, arg_dtype in enumerate(sig.args):
            if arg_dtype.const:
                arguments.append(self._literal(arg_dtype))
                continue

            # The first argument in an expression can't be a literal
            arg = self._expression(
                arg_dtype,
                arg_depth[arg_i],
                can_be_literal=(arg_i != 0),
                can_be_window=can_be_window,
                can_be_aggregate=can_be_aggregate,
            )
            arguments.append(arg)

        if sig.is_vararg:
            num_varargs = self.random.randint(0, 5)
            vararg_depth = [
                self.random.randint(1, depth - 1) for _ in range(num_varargs)
            ]
            vararg_dtype = sig.args[-1]
            for i in range(num_varargs):
                if not vararg_dtype.const:
                    vararg = self._expression(
                        vararg_dtype,
                        vararg_depth[i],
                        can_be_window=can_be_window,
                        can_be_aggregate=can_be_aggregate,
                    )
                else:
                    vararg = self._literal(vararg_dtype)
                arguments.append(vararg)

        context_kwargs = self._context_kwargs(op, sig, dtype, depth)
        return exp.SymbolicExpression(
            exp.FunctionCall(op.name, *arguments, **context_kwargs)
        )

    def _context_kwargs(
        self,
        op: ops.Operator,
        sig: OperatorSignature,
        dtype: dtypes.DType,
        depth: int,
    ) -> dict:
        return {}

    def _special_case(
        self,
        op: ops.Operator,
        sig: OperatorSignature,
        dtype: dtypes.DType,
        depth: int,
    ) -> exp.FunctionCall | None:
        return None

    def _literal(self, dtype: dtypes.DType):
        if isinstance(dtype, dtypes.NoneDType):
            return None
        return self.random.choice(self.literals[dtype.without_modifiers()])

    def _column(self, dtype: dtypes.DType):
        return self.random.choice(self.cols[dtype.without_modifiers()])

    def _dtype(self):
        return self.random.choice(self.available_dtypes)

    # Overrideable Methods

    def _filter_op(self, op: ops.Operator, signatures: list[OperatorSignature]) -> bool:
        if isinstance(op, ops.Marker):
            return False
        return True


class ExpressionFuzzer(BaseExpressionFuzzer):
    def __init__(
        self,
        table: Table,
        seed: int = 0xBADC0DE,
    ):
        super().__init__(table, seed)
        self.arrange_fuzzer = ArrangeExpressionFuzzer(table=table, seed=(seed ^ 0xDA9))

    def _context_kwargs(
        self,
        op: ops.Operator,
        sig: OperatorSignature,
        dtype: dtypes.DType,
        depth: int,
    ) -> dict:
        context_kwargs = {}
        if op.context_kwargs is None:
            return context_kwargs

        if "arrange" in op.context_kwargs and not isinstance(
            op, ops.window.WindowImplicitArrange
        ):
            arrange = self.arrange_fuzzer.generate_expression(max_depth=3)
            context_kwargs["arrange"] = [arrange]

        return context_kwargs

    def _special_case(
        self,
        op: ops.Operator,
        sig: OperatorSignature,
        dtype: dtypes.DType,
        depth: int,
    ) -> exp.FunctionCall | None:
        if isinstance(op, ops.window.WindowImplicitArrange):
            # Rank, DenseRank, etc.
            arrange = self.arrange_fuzzer.generate_expression(dtype=dtype, max_depth=3)
            return exp.FunctionCall(op.name, arrange)

        return None


class ArrangeExpressionFuzzer(BaseExpressionFuzzer):
    def generate_expression(self, *args, **kwargs):
        kwargs["can_be_literal"] = False
        expr = super().generate_expression(*args, **kwargs)

        first_last = self.random.choice((ops.NullsFirst, ops.NullsLast))
        pos_neg = self.random.choice((ops.Pos, ops.Neg))

        expr = exp.FunctionCall(first_last.name, expr)
        expr = exp.FunctionCall(pos_neg.name, expr)

        return exp.SymbolicExpression(expr)

    def _filter_op(self, op: ops.Operator, signatures: list[OperatorSignature]) -> bool:
        if isinstance(op, (ops.Window, ops.Aggregate)):
            return False
        if isinstance(op, ops.Marker):
            return False

        return True
