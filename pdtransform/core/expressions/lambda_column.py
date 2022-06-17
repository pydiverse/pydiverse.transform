from .expression import SymbolicExpression


class LambdaColumnGetter:
    """
    An instance of this object can be used to instantiate a LambdaColumn.
    """
    def __getattr__(self, item):
        return LambdaColumn(item)

    def __getitem__(self, item):
        return LambdaColumn(item)


class LambdaColumn(SymbolicExpression):
    """ Anonymous Column

    A lambda column is a column without an associated table or UUID. It is just
    a name that inherits from `SymbolicExpression`. This means that it can be
    used to reference columns in the same pipe as it was created.

    Example:
      The following fails because `table.a` gets referenced before it gets created.
        table >> mutate(a = table.x) >> mutate(b = table.a)
      Instead you can use a lambda column to achieve this:
        table >> mutate(a = table.x) >> mutate(b = λ.a)
    """
    __slots__ = ('_name', )
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f'<λ.{self._name}>'

    def __hash__(self):
        return hash(self._name)


# Global instance of LambdaColumnGetter.
λ = LambdaColumnGetter()