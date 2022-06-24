from collections import ChainMap
from typing import Any, Iterable

from .column import Column, LambdaColumn, generate_col_uuid
from .dispatchers import builtin_verb
from .expressions import FunctionCall
from .expressions import SymbolicExpression
from .expressions.utils import iterate_over_expr
from .table_impl import AbstractTableImpl
from .utils import ordered_set, bidict


def check_cols_available(tables: AbstractTableImpl | Iterable[AbstractTableImpl], columns: set[Column], function_name: str):
    if isinstance(tables, AbstractTableImpl):
        tables = (tables, )
    available_columns = ChainMap(*(table.available_cols for table in tables))
    missing_columns = []
    for col in columns:
        if col.uuid not in available_columns:
            missing_columns.append(col)
    if missing_columns:
        missing_columns_str = ", ".join(map(lambda x: str(x), missing_columns))
        raise ValueError(f"Can't access column(s) {missing_columns_str} in {function_name}() because they aren't available in the input.")

def check_lambdas_valid(tbl: AbstractTableImpl, *expressions):
    lambdas = []
    for expression in expressions:
        lambdas.extend(l for l in iterate_over_expr(expression) if isinstance(l, LambdaColumn))
    missing_lambdas = { l for l in lambdas if l.name not in tbl.named_cols.fwd }
    if missing_lambdas:
        missing_lambdas_str = ", ".join(map(lambda x: str(x), missing_lambdas))
        raise ValueError(f"Invalid lambda column(s) {missing_lambdas_str}.")

def cols_in_expression(expression) -> set[Column]:
    return { c for c in iterate_over_expr(expression) if isinstance(c, Column)}

def cols_in_expressions(expressions) -> set[Column]:
    if len(expressions) == 0:
        return set()
    return set.union(*(cols_in_expression(e) for e in expressions))


@builtin_verb()
def alias(tbl: AbstractTableImpl, name: str):
    """Creates a new table object with a different name."""
    return tbl.alias(name)

@builtin_verb()
def collect(tbl: AbstractTableImpl):
    return tbl.collect()

@builtin_verb()
def select(tbl: AbstractTableImpl, *args: Column | LambdaColumn):
    # TODO: Allow ore complex expressions in args (eg not expressions)
    # >> select(...)    ->   Select all columns
    if len(args) == 1 and args[0] is Ellipsis:
        raise NotImplementedError

    # Validate input
    check_cols_available(tbl, cols_in_expressions(args), 'select')
    check_lambdas_valid(tbl, *args)

    for col in args:
        if not isinstance(col, (Column, LambdaColumn)):
            raise ValueError(f"Arguments to select verb must be of type 'Column' or 'LambdaColumn' and not '{type(col)}'.")

    selects = []
    for col in args:
        if isinstance(col, Column):
            selects.append(tbl.named_cols.bwd[col.uuid])
        elif isinstance(col, LambdaColumn):
            selects.append(col.name)
    # SELECT
    new_tbl = tbl.copy()
    new_tbl.selects = ordered_set(selects)
    new_tbl.select(*args)
    return new_tbl

@builtin_verb()
def mutate(tbl: AbstractTableImpl, **kwargs: SymbolicExpression):
    # Check args only contains valid columns
    check_cols_available(tbl, cols_in_expressions(kwargs.values()), 'mutate')
    kwargs = {k: tbl.resolve_lambda_cols(v) for k, v in kwargs.items()}

    new_tbl = tbl.copy()
    for name, expr in kwargs.items():
        uid = generate_col_uuid()
        new_tbl.selects.add(name)
        new_tbl.named_cols.fwd[name] = uid
        new_tbl.available_cols.add(uid)
        new_tbl.col_expr[uid] = expr

    new_tbl.mutate(**kwargs)
    return new_tbl

@builtin_verb()
def join(left: AbstractTableImpl, right: AbstractTableImpl, on: SymbolicExpression, how: str):
    # TODO: Also allow on to be a dictionary
    # Check args only contains valid columns
    check_cols_available((left, right), cols_in_expression(on), 'join')

    if how not in ('inner', 'left', 'outer'):
        raise ValueError(f"Join type must be one of 'inner', 'left' or 'outer' (value provided: {how=})")

    # Construct new table
    # JOIN -> Merge the left and right table
    new_left = left.copy()

    # Update selects
    right_renamed_selects = ordered_set(name + '_' + right.name for name in right.selects)
    right_renamed_cols = { k + '_' + right.name: v for k, v in right.named_cols.fwd.items() }

    # Check for collisions
    # TODO: Review...
    if ambiguous_column_names := new_left.selects & right_renamed_selects:
        raise ValueError('Ambiguous column names: ' + ', '.join(ambiguous_column_names))
    if ambiguous_column_names := set(new_left.named_cols.fwd.keys()) & right_renamed_cols.keys():
        raise ValueError('Ambiguous column names: ' + ', '.join(ambiguous_column_names))
    if ambiguous_column_uuids := set(new_left.named_cols.fwd.values()) & set(right_renamed_cols.values()):
        raise ValueError('Ambiguous column uuids: ' + ', '.join(map(str, ambiguous_column_uuids)))

    new_left.selects |= right_renamed_selects
    new_left.named_cols.fwd.update(right_renamed_cols)
    new_left.available_cols.update(right.available_cols)
    new_left.col_expr.update(right.col_expr)
    new_left.col_dtype.update(right.col_dtype)

    # By resolving lambdas this late, we enable the user to use lambda columns
    # to reference mutated columns from the right side of the join.
    # -> `λ.columnname_righttablename` is a valid lambda in the on condition.
    check_lambdas_valid(new_left, on)
    on = new_left.resolve_lambda_cols(on)

    new_left.join(right, on, how)
    return new_left

@builtin_verb()
def filter(tbl: AbstractTableImpl, *args: SymbolicExpression):
    # TODO: Type check expression
    check_cols_available(tbl, cols_in_expressions(args), 'filter')
    args = [tbl.resolve_lambda_cols(arg) for arg in args]

    new_tbl = tbl.copy()
    new_tbl.filter(*args)
    return new_tbl

@builtin_verb()
def arrange(tbl: AbstractTableImpl, *args: Column | LambdaColumn):
    # Validate Input
    check_cols_available(tbl, cols_in_expressions(args), 'arrange')
    check_lambdas_valid(tbl, *args)

    # Determine if ascending or descending
    def ordering_pealer(expr: Any):
        num_neg = 0
        while isinstance(expr, FunctionCall) and expr.operator == '__neg__':
            num_neg += 1
            expr = expr.args[0]
        return expr, num_neg % 2 == 0

    ordering = []
    for arg in args:
        col, ascending = ordering_pealer(arg)
        if not isinstance(col, (Column, LambdaColumn)):
            raise ValueError(f"Arguments to select verb must be of type 'Column' or 'LambdaColumn' and not '{type(col)}'.")
        col = tbl.resolve_lambda_cols(col)
        ordering.append((col, ascending))

    new_tbl = tbl.copy()
    new_tbl.arrange(ordering)
    return new_tbl

@builtin_verb()
def group_by(tbl: AbstractTableImpl, *args: Column | LambdaColumn, add = False):
    # Validate Input
    check_cols_available(tbl, cols_in_expressions(args), 'group_by')
    check_lambdas_valid(tbl, *args)

    # WARNING: Depending on the SQL backend, you might only be allowed to reference columns
    if not args:
        raise ValueError("Expected columns to group by, but none were specified. To remove the grouping use the ungroup verb instead.")
    for col in args:
        if not isinstance(col, (Column, LambdaColumn)):
            raise ValueError(f"Arguments to group_by verb must be of type 'Column' or 'LambdaColumn' and not '{type(col)}'.")

    args = [tbl.resolve_lambda_cols(arg) for arg in args]

    new_tbl = tbl.copy()
    if add:
        new_tbl.grouped_by |= ordered_set(args)
    else:
        new_tbl.grouped_by = ordered_set(args)
    new_tbl.group_by(*args)
    return new_tbl

@builtin_verb()
def ungroup(tbl: AbstractTableImpl):
    """ Remove all groupings from table. """
    new_tbl = tbl.copy()
    new_tbl.grouped_by.clear()
    new_tbl.ungroup()
    return new_tbl

@builtin_verb()
def summarise(tbl: AbstractTableImpl, **kwargs: SymbolicExpression):
    # Validate Input
    check_cols_available(tbl, cols_in_expressions(kwargs.values()), 'summarise')

    new_tbl = tbl.copy()
    new_tbl.pre_summarise()

    kwargs = {k: new_tbl.resolve_lambda_cols(v) for k, v in kwargs.items()}

    # TODO: Validate that the functions are actually aggregating functions.
    ...

    # Calculate state for new table
    selects = ordered_set()
    named_cols = bidict()
    available_cols = set()
    col_expr = {}

    # Add grouping cols to beginning of select.
    for col in tbl.grouped_by:
        selects.add(tbl.named_cols.bwd[col.uuid])
        available_cols.add(col.uuid)
        named_cols.fwd[col.name] = col.uuid

    # Add summarizing cols to the end of the select.
    for name, expr in kwargs.items():
        if name in selects:
            raise ValueError(f"Column with name '{name}' already in select. The new summarised columns must have a different name than the grouping columns.")
        uid = generate_col_uuid()
        selects.add(name)
        named_cols.fwd[name] = uid
        available_cols.add(uid)
        col_expr[uid] = expr

    # Update new_tbl
    new_tbl.selects = ordered_set(selects)
    new_tbl.named_cols = named_cols
    new_tbl.available_cols = available_cols
    new_tbl.col_expr.update(col_expr)
    new_tbl.intrinsic_grouped_by = new_tbl.grouped_by.copy()
    new_tbl.summarise(**kwargs)

    # Reduce the grouping level by one -> drop last
    if len(new_tbl.grouped_by):
        new_tbl.grouped_by.pop_back()

    if len(new_tbl.grouped_by):
        new_tbl.group_by(*new_tbl.grouped_by)
    else:
        new_tbl.ungroup()

    return new_tbl