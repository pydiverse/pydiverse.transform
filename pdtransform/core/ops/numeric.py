from . import ElementWise, Unary, Binary


__all__ = [
    'Add', 'RAdd',
    'Sub', 'RSub',
    'Mul', 'RMul',
    'TrueDiv', 'RTrueDiv',
    'FloorDiv', 'RFloorDiv',
    'Pow', 'RPow',
    'Mod', 'RMod',
    'Neg', 'Pos',
    'Round',
]


class Numeric(ElementWise):
    def validate_signature(self, signature):
        numeric_types = ('int', 'float')
        assert (all((arg in numeric_types) for arg in signature.args)
                and signature.rtype in numeric_types)
        super().validate_signature(signature)


class Add(Numeric, Binary):
    name = '__add__'
    signatures = [
        'int, int -> int',
        'int, float -> float',
        'float, int -> float',
        'float, float -> float',
    ]


class RAdd(Add):
    name = '__radd__'


class Sub(Numeric, Binary):
    name = '__sub__'
    signatures = [
        'int, int -> int',
        'int, float -> float',
        'float, int -> float',
        'float, float -> float',
    ]


class RSub(Sub):
    name = '__rsub__'


class Mul(Numeric, Binary):
    name = '__mul__'
    signatures = [
        'int, int -> int',
        'int, float -> float',
        'float, int -> float',
        'float, float -> float',
    ]


class RMul(Mul):
    name = '__rmul__'


class TrueDiv(Numeric, Binary):
    name = '__truediv__'
    signatures = [
        'int, int -> float',
        'int, float -> float',
        'float, int -> float',
        'float, float -> float',
    ]


class RTrueDiv(TrueDiv):
    name = '__rtruediv__'


class FloorDiv(Numeric, Binary):
    name = '__floordiv__'
    signatures = [
        'int, int -> int',
    ]


class RFloorDiv(FloorDiv):
    name = '__rfloordiv__'


class Pow(Numeric, Binary):
    name = '__pow__'
    signatures = [
        'int, int -> int',
    ]


class RPow(Pow):
    name = '__rpow__'


class Mod(Numeric, Binary):
    name = '__mod__'
    signatures = [
        'int, int -> int',
    ]


class RMod(Mod):
    name = '__rmod__'


class Neg(Numeric, Unary):
    name = '__neg__'
    signatures = [
        'int -> int',
        'float -> float',
    ]


class Pos(Numeric, Unary):
    name = '__pos__'
    signatures = [
        'int -> int',
        'float -> float',
    ]


class Round(Numeric):
    name = '__round__'
    signatures = [
        'int -> int',
        'int, int -> int',
        'float -> int',
        'float, int -> float',
    ]
