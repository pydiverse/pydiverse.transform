from .column import Column, generate_col_uuid
from .dispatchers import builtin_verb
from .expressions import SymbolicExpression
from .expressions.expression import iterate_over_expr
from .expressions.lambda_column import LambdaColumn
from .table_impl import AbstractTableImpl


def check_is_cols_subset(superset: set[Column], subset: set[Column], function_name: str):
    if subset.issubset(superset):
        return
    missing_columns = subset - superset
    missing_columns_str = ", ".join(map(lambda x: str(x), missing_columns))
    raise ValueError(f"Can't access column(s) {missing_columns_str} in {function_name}() because they aren't avaiable in the input.")

def check_lambdas_valid(tbl: AbstractTableImpl, *expressions):
    lambdas = []
    for expression in expressions:
        lambdas.extend(l for l in iterate_over_expr(expression) if isinstance(l, LambdaColumn))
    missing_lambdas = { l for l in lambdas if l._name not in tbl.named_cols.fwd }
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
def collect(tbl: AbstractTableImpl):
    return tbl.collect()

@builtin_verb()
def select(tbl: AbstractTableImpl, *args: Column | LambdaColumn):
    # TODO: Allow ore complex expressions in args (eg not expressions)
    # >> select(...)    ->   Select all columns
    if len(args) == 1 and args[0] is Ellipsis:
        raise NotImplementedError

    # Validate input
    check_is_cols_subset(tbl.available_columns, cols_in_expressions(args), 'select')
    check_lambdas_valid(tbl, *args)

    for col in args:
        if not isinstance(col, (Column, LambdaColumn)):
            raise ValueError(f"Arguments to select verb must be of type 'Column' or 'LambdaColumn' and not '{type(col)}'.")

    selects = []
    for col in args:
        if isinstance(col, Column):
            selects.append(tbl.named_cols.bwd[col._uuid])
        elif isinstance(col, LambdaColumn):
            selects.append(col._name)
    # SELECT
    new_tbl = tbl.copy()
    new_tbl.selects = { name: None for name in selects }
    new_tbl.select(*args)
    return new_tbl

@builtin_verb()
def mutate(tbl: AbstractTableImpl, **kwargs: SymbolicExpression):
    # Check args only contains valid columns
    check_is_cols_subset(tbl.available_columns, cols_in_expressions(kwargs.values()), 'mutate')
    kwargs = {k: tbl.resolve_lambda_cols(v) for k, v in kwargs.items()}

    new_tbl = tbl.copy()
    for name, expr in kwargs.items():
        uid = generate_col_uuid()
        new_tbl.selects[name] = None
        new_tbl.named_cols.fwd[name] = uid
        new_tbl.col_expr[uid] = expr

    new_tbl.mutate(**kwargs)
    return new_tbl

@builtin_verb()
def join(left: AbstractTableImpl, right: AbstractTableImpl, on: SymbolicExpression, how: str):
    # TODO: Also allow on to be a dictionary
    # Check args only contains valid columns
    on_cols = cols_in_expression(on)
    available_cols = left.available_columns | right.available_columns
    check_is_cols_subset(available_cols, on_cols, 'join')

    on = left.resolve_lambda_cols(on)

    # Check for name collisions
    if ambiguous_cols := ({str(c) for c in left.available_columns}
                        & {str(c) for c in right.available_columns}):
        ambiguous_cols_str = ', '.join(ambiguous_cols)
        raise ValueError(f'Ambiguous column name(s): {ambiguous_cols_str}. '
                         'Make sure that all tables in the join have unique names.')

    if how not in ('inner', 'left', 'outer'):
        raise ValueError(f"Join type must be one of 'inner', 'left' or 'outer' (value provided: {how=})")

    # Construct new table
    # JOIN -> Merge the left and right table
    new_left = left.copy()

    # Update selects
    # TODO: Find a proper renaming scheme
    right_renamed_selects = { right.name + '_' + k: v for k, v in right.selects.items() }
    right_renamed_cols = { right.name + '_' + k: v for k, v in right.named_cols.fwd.items() }

    for k, v in right_renamed_selects.items():
        if k in new_left.selects:
            raise Exception
        new_left.selects[k] = v

    for k,v in right_renamed_cols.items():
        if k in new_left.named_cols.fwd:
            raise Exception
        new_left.named_cols.fwd[k] = v

    new_left.col_expr.update(right.col_expr)
    new_left.col_dtype.update(right.col_dtype)
    new_left.available_columns.update(right.available_columns)

    new_left.join(right, on, how)
    return new_left

@builtin_verb()
def filter(tbl: AbstractTableImpl, *args: SymbolicExpression):
    # TODO: Type check expression
    condition_cols = cols_in_expressions(args)
    check_is_cols_subset(tbl.available_columns, condition_cols, 'filter')
    args = [tbl.resolve_lambda_cols(arg) for arg in args]

    new_tbl = tbl.copy()
    new_tbl.filter(*args)
    return new_tbl
