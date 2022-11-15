=======
Plugins
=======

Usage
-----
Starting in version 0.5.0, Picasso allows for creating plugins. These can be used only after Picasso installation via GitHub, see `here <https://github.com/jungmannlab/picasso#:~:text=Via%20GitHub%3A,picasso%20localize.>`_.

To use a given plugin, simply copy it to your Picasso direcotory in ``picasso/gui/plugins``. Please keep in mind that the ``__init__.py`` file in the ``plugins`` folder must not be modified or deleted.

For developers
--------------
To create a plugin, you can use the template provided in ``picasso/plugin_template.py``, see below.

.. image:: ../docs/plugins.png
   :scale: 70 %
   :alt: Plugins

As an example, please see any of the plugins on the GitHub `repo <https://github.com/rafalkowalewski1/picasso_plugins>`_.