from __future__ import annotations

import pandas as pd
import sqlalchemy as sa

from pydiverse.transform import Table
from pydiverse.transform.core.verbs import *
from pydiverse.transform.eager import PandasTableImpl
from pydiverse.transform.lazy import SQLTableImpl


def test_example():
    dfA = pd.DataFrame(
        {
            "x": [1],
            "y": [2],
        }
    )
    dfB = pd.DataFrame(
        {
            "a": [2, 1, 0, 1],
            "x": [1, 1, 2, 2],
        }
    )

    input1 = Table(PandasTableImpl("dfA", dfA))
    input2 = Table(PandasTableImpl("dfB", dfB))

    transform = (
        input1
        >> left_join(input2 >> select(), input1.x == input2.x)
        >> mutate(x5=input1.x * 5, a=input2.a)
    )
    out1 = transform >> collect()
    print("\nPandas based result:")
    print(out1)

    engine = sa.create_engine("sqlite:///:memory:")
    dfA.to_sql("dfA", engine, index=False, if_exists="replace")
    dfB.to_sql("dfB", engine, index=False, if_exists="replace")
    input1 = Table(SQLTableImpl(engine, "dfA"))
    input2 = Table(SQLTableImpl(engine, "dfB"))
    transform = (
        input1
        >> left_join(input2 >> select(), input1.x == input2.x)
        >> mutate(x5=input1.x * 5, a=input2.a)
    )
    out2 = transform >> collect()
    print("\nSQL query:")
    print(transform >> build_query())
    print("\nSQL based result:")
    print(out2)

    out1 = out1.sort_values("a").reset_index(drop=True)
    out2 = out2.sort_values("a").reset_index(drop=True)
    assert len(out1.compare(out2)) == 0
