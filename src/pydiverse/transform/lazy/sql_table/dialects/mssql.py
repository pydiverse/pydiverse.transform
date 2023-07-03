from pydiverse.transform.lazy.sql_table.sql_table import SQLTableImpl


class MSSqlTableImpl(SQLTableImpl):
    _dialect_name = "mssql"
