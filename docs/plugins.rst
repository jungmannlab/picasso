=======
Plugins
=======

Usage
-----
A plugin is a single Python (``.py``) file that adds new actions to a Picasso GUI.

Installing plugins
~~~~~~~~~~~~~~~~~~~
Place your plugin ``.py`` file(s) in the user plugins folder:

- ``~/.picasso/plugins`` (on Windows this is ``C:\Users\<your user name>\.picasso\plugins``).

This folder is created automatically the first time you run any Picasso GUI, and the **same folder is used for every installation type** (one-click installer, PyPI, conda or GitHub). The easiest way to open it is the **Plugins → Open plugins folder…** menu entry available inside any Picasso app. After copying a plugin there, use **Plugins → Reload plugins** (or restart the app) to load it.

Because the plugins folder lives in your home directory and not inside the Picasso installation, plugins are kept when you update Picasso and are not left behind when you uninstall it.

**NOTE**: With the one-click installer, plugins can only use packages that are
installed with Picasso (the dependencies listed in ``pyproject.toml``).

For developers
--------------
To create a plugin, you can use the template provided in `picasso/plugin_template.py <https://github.com/jungmannlab/picasso/blob/master/plugin_template.py>`_. For more examples of plugins, please see the `GitHub repo <https://github.com/rafalkowalewski1/picasso_plugins>`_.
