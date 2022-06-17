from pdtransform.core.dispatchers import builtin_verb
from .lazy_table import LazyTableImpl


@builtin_verb(LazyTableImpl)
def show_query(tbl: LazyTableImpl):
    print(tbl.query_string())
    return tbl