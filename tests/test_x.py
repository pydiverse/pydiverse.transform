import pandas as pd
import pytest
import sqlalchemy

from pdtransform import λ
from pdtransform.core.dispatchers import inverse_partial, verb
from pdtransform.core.table import Table
from pdtransform.core.verbs import collect, select, mutate, join, filter
from pdtransform.eager.pandas_table import PandasTableImpl
from pdtransform.lazy.sql_table import SQLTableImpl
from pdtransform.lazy.verbs import show_query

dfA = pd.DataFrame({
    "name":  ["a", "b", "c", "d", "e", "f", "g"],
    "value": [1, 2, 3, 4, 5, 6, 7],
})
dfB = pd.DataFrame({
    "name": ["a", "b", "a", "b"],
    "c": [1, 20, 3, 40],
})
engine = sqlalchemy.create_engine('sqlite:///:memory:')
dfA.to_sql('dfA', engine, index = False, if_exists = 'replace')
dfB.to_sql('dfB', engine, index = False, if_exists = 'replace')


def test_join():
    tA = Table(SQLTableImpl(engine, 'dfA'))
    tB = Table(SQLTableImpl(engine, 'dfB'))
    # tA = Table(PandasTableImpl('dfA', dfA))
    # tB = Table(PandasTableImpl('dfB', dfB))

    print(
        tA
        >> select()
        >> join(tB, (tA.name == tB.name), 'left')
        # >> select(tB.c)
        # >> mutate(dfB_name = tA.name, name = tB.name)
        # >> filter(tA.value >= 2)
        # >> filter(tA.value < 4)
        >> collect()
    )

def test_str_trim():
    df = pd.DataFrame({
        'a': [' x ', '  y ', 'z']
    })
    tbl = Table(PandasTableImpl('tbl', df))
    print(tbl >> mutate(b = tbl.a.str.strip()) >> collect())