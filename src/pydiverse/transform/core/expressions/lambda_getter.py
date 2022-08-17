from __future__ import annotations

from pydiverse.transform.core.column import LambdaColumn
from pydiverse.transform.core.expressions import SymbolicExpression


class LambdaColumnGetter:
    """
    An instance of this object can be used to instantiate a LambdaColumn.
    """

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{item}'"
            )
        return SymbolicExpression(LambdaColumn(item))

    def __getitem__(self, item):
        return SymbolicExpression(LambdaColumn(item))


# Global instance of LambdaColumnGetter.
λ = LambdaColumnGetter()
