{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

.. currentmodule:: {{ fullname }}

{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if classes %}
.. rubric:: Classes

.. autosummary::
   :nosignatures:
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if functions %}
.. rubric:: Functions

.. autosummary::
   :nosignatures:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% for item in classes %}
.. autoclass:: {{ fullname }}.{{ item }}
   :members:
   :show-inheritance:
{% endfor %}

{% for item in functions %}
.. autofunction:: {{ fullname }}.{{ item }}
{% endfor %}
