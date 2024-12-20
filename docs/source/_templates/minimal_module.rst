{% set name = fullname.split('.')[-1] %}
{{ name | escape | underline}}

.. automodule:: {{ fullname }}

   {% block docstring %}
   {% endblock %}
