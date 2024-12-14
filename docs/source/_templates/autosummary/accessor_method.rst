{{ name }}
{{ underline }}

.. currentmodule:: {{ '.'.join(module.split('.')[:2]) }}

.. autoaccessormethod:: {{ (module.split('.')[2:] + [objname]) | join('.') }}
