{{ '.'.join(objname.split('.')[-2:]) }}
{{ underline }}

.. currentmodule:: {{ '.'.join(module.split('.')[:2]) }}

.. autoaccessormethod:: {{ (module.split('.')[2:] + [objname]) | join('.') }}
