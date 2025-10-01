=======
Plugins
=======

Usage
-----
Starting in version 0.5.0, Picasso supports plugins. Below are the instructions on how to install them.

*Keep in mind that the* ``__init__.py`` *file in the* ``picasso/picasso/gui/plugins`` *folder must not be modified or deleted.*

One click installer
~~~~~~~~~~~~~~~~~~~
**NOTE**: After uninstalling Picasso, ``Picasso`` folder needs to be deleted manually, as the uninstaller currently does not remove the plugins automatically.

- Find the location where you installed Picasso. By default, it is ``C:/Picasso``. *Before version 0.8.3, the default location was* ``C:/Program Files/Picasso``.
- Then go to the folder ``picasso/gui/plugins``.
- Copy the plugin(s) to this folder.

**NOTE**: Plugins added in this distribution will not be able to use packages that are not installed automatically (from the file ``requirements.txt``).

PyPI
~~~~
If you installed Picasso using ``pip install picassosr``, you can add plugins by following these steps:

- Activate your conda environment where ``picassosr`` is installed by typing ``conda activate YOUR_ENVIRONMENT``.
- To find the location of the package, type ``pip show picassosr`` and look for the line starting with ``Location:``.
- Navigate to this location and go to ``picasso/gui/plugins``.
- Copy the plugin(s) to this folder.

GitHub
~~~~~~
If you cloned the GitHub repository, you can add plugins by following these steps:
- Find the directory where you cloned the GitHub repository with Picasso.
- Go to ``picasso/picasso/gui/plugins``.
- Copy the plugin(s) to this folder.

For developers
--------------
To create a plugin, you can use the template provided in ``picasso/plugin_template.py``, see below.

.. image:: ../docs/plugins.png
   :scale: 70 %
   :alt: Plugins

As an example, please see any of the plugins in the `GitHub repo <https://github.com/rafalkowalewski1/picasso_plugins>`_.