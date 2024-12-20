{% set name = fullname.split('.')[-1] %}
{{ name | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
      .. rubric:: {{ _('Module Attributes') }}

      .. autosummary::
         :toctree:
         :template: attribute.rst
      {% for item in attributes %}
         {{ item }}
      {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
      .. rubric:: {{ _('Functions') }}

      .. autosummary::
         :toctree:
         :template: method.rst
      {% for item in functions %}
         {{ item }}
      {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
      .. rubric:: {{ _('Classes') }}

      .. autosummary::
         :toctree:
         :template: class.rst
      {% for item in classes %}
         {{ item }}
      {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: {{ _('Modules') }}

.. autosummary::
   :toctree:
   :template: module.rst
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}