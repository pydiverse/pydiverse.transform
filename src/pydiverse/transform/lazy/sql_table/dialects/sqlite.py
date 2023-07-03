from pydiverse.transform.lazy.sql_table.sql_table import SQLTableImpl


class SQLiteTableImpl(SQLTableImpl):
    _dialect_name = "sqlite"
