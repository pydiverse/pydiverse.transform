from .core import ElementWise, Unary


__all__ = [
    'Strip',
]


class StringUnary(ElementWise, Unary):
    signatures = [
        'str -> str',
    ]


class Strip(StringUnary):
    name = 'strip'
