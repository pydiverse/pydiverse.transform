{% if (name == "__add__") %}\+
{% elif name == "__sub__" %}\-
{% elif name == "__mul__" %}\*
{% elif name == "__truediv__" %}\/
{% elif name == "__floordiv__" %}\/\/
{% elif name == "__pow__" %}\*\* (pow)
{% elif name == "__mod__" %}\%
{% elif name == "__pos__" %}\+ (unary)
{% elif name == "__neg__" %}\- (unary)
{% elif name == "__lt__" %}\<
{% elif name == "__le__" %}\<\=
{% elif name == "__gt__" %}\>
{% elif name == "__ge__" %}\>\=
{% elif name == "__eq__" %}\=\=
{% elif name == "__ne__" %}\!\=
{% elif name == "__or__" %}\|
{% elif name == "__and__" %}\&
{% elif name == "__xor__" %}\^
{% elif name == "__invert__" %}\~
{% else %}{{ name }}
{% endif %}{{ underline }}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
