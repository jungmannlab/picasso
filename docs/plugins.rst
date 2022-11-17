=======
Plugins
=======

Usage
-----
Starting in version 0.5.0, Picasso allows for creating plugins. They can be added by the user for each of the available distributions of picasso (GitHub, picassosr from PyPI and the one click installer). However, using plugins in the latter may cause issues, see below.

Please keep in mind that the ``__init__.py`` file in the ``plugins`` folder must not be modified or deleted.

GitHub
~~~~~~
- Find the directory where you cloned the GitHub repository with Picasso.
- Go to ``picasso/picasso/gui/plugins``.
- Copy the plugin(s) to this folder.
- The plugin(s) should automatically work after running picasso in the new command window.

PyPI
~~~~
- Find the location of the environment where picassosr is installed (type ``conda env list`` to see the directory).
- Find this directory and go to ``YOUR_ENVIRONMENT/Lib/site-packages/picasso/gui/plugins``.
- Copy the plugin(s) to this folder.
- The plugin(s) should automatically work after running picasso in the new command window.

One click installer
~~~~~~~~~~~~~~~~~~~
**NOTE**: This may lead to issues if Picasso is installed, as the plugin scripts will remain in the ``plugins`` folder upon deinstallation. After deinstalltion, the ``Picasso`` folder needs to be deleted manually.

- Find the location where you installed Picasso. By default, it is ``C:/Program Files/Picasso``.
- Go to the following subfolder in the `Picasso` directory: ``picasso/gui/plugins``.
- Copy the plugin(s) to this folder.
- The plugin(s) should automatically be loaded after double-clicking the respective desktop shortcuts.

**NOTE**: Plugins added in this distribution will not be able to use packages that are not installed automatically (from the file ``requirements.txt``).

For developers
--------------
To create a plugin, you can use the template provided in ``picasso/plugin_template.py``, see below.

.. image:: ../docs/plugins.png
   :scale: 70 %
   :alt: Plugins

As an example, please see any of the plugins on the GitHub `repo <https://github.com/rafalkowalewski1/picasso_plugins>`_.