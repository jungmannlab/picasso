Plugins
=======

.. _plugins-usage:

Usage
-----
A plugin is a single Python (``.py``) file that extends Picasso. It can add new
actions to a Picasso GUI, new ``picasso`` command line subcommands, new
functions importable from your own scripts, or any combination of the three.

.. warning::

   A plugin is ordinary Python code running inside Picasso, so an enabled
   plugin can do anything Picasso itself can do: read and write your files,
   use the network, start other programs. Picasso cannot sandbox plugins.

- Plugins are **not loaded until you enable them**, so copying a file into the
  plugins folder is not enough to make it run;
- Downloads from the online registry are **checked against the SHA-256 hash
  published in the registry**, so the code behind a plugin cannot change
  without a visible, reviewed change to the registry;
- The plugin browser can **show you a plugin's source** before you install it,
  and a **diff of what changed** before you update it.

.. _plugins-registry:

Installing plugins from the online registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The easiest way to get plugins is the built-in store: open
``Plugins > Browse online plugins...`` in any Picasso app.

- It lists the plugins published in the
  `picasso_plugins registry <https://github.com/jungmannlab/picasso_plugins>`_,
  filtered to the app you are using (tick *Show plugins for all Picasso apps*
  to see the rest).
- It lets you **Install**, **Update** or **Uninstall** each one with a single
  click. Installed plugins take effect immediately - no restart needed.

Each row also offers **View source** (**View changes** for an update), which
downloads the file, verifies it against the registry hash and shows it
read-only. Most plugins are short; reading one before enabling it is
realistic and is the only check that actually tells you what a plugin does.

The instructions on how to use a plugin are written in the docstring at the
very top of its ``.py`` file, so **View source** is also where you read them:
the first block of text you see explains what the plugin does and how to use
it.

A plugin you install this way is enabled automatically - clicking *Install*
and confirming the warning *is* the consent step. Picasso shows that warning
once, before your first install.

If a plugin is listed as **No integrity hash**, the registry does not publish
a SHA-256 for it and Picasso will refuse to install it. That is a problem with
the registry entry, not with your setup; please report it in the
`picasso_plugins repository <https://github.com/jungmannlab/picasso_plugins>`_.

.. _plugins-manual-install:

Installing plugins manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Place your plugin ``.py`` file(s) in the user plugins folder:

- ``~/.picasso/plugins`` (on Windows this is
  ``C:\Users\<your user name>\.picasso\plugins``).

This folder is created automatically the first time you run any Picasso GUI,
and the **same folder is used for every installation type** (one-click
installer, PyPI, conda or GitHub). The easiest way to open it is the
``Plugins > Open plugins folder...`` menu entry available inside any Picasso
app.

A file copied in by hand is **found but not enabled**:

1. It shows up in ``Plugins > Browse online plugins...`` as a *Local file*,
   and the Plugins menu tells you how many such files are waiting.
2. To start using it, review it with **View source**, then tick its
   **Enabled** checkbox and confirm the warning.
3. Untick the checkbox at any time to stop loading it without deleting the
   file.

Because Picasso cannot verify a file that did not come from the registry, this
is the least safe way to share a plugin. If you are sharing your own plugin
with others, publishing it in the
`registry <https://github.com/jungmannlab/picasso_plugins>`_ is better for
everyone: recipients get a hash-verified download, a version number and
one-click updates, and the code is reviewed in the open before it reaches
anyone.

.. note::

   With the one-click installer, plugins can only use packages that are
   installed with Picasso (the dependencies listed in ``pyproject.toml``).

.. _plugins-without-gui:

Managing plugins without a GUI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Everything the plugin browser does is also available from the command line
(see :ref:`cmd-plugins`), so plugins can be used on a headless machine, in a
batch script or over SSH, and set up without opening a Picasso app:

.. code-block:: bash

   picasso plugins list                  # what is there, and what each one provides
   picasso plugins path                  # print the plugins folder
   picasso plugins enable my_plugin.py   # asks you to confirm first
   picasso plugins disable my_plugin.py  # stop loading it, without deleting it

   picasso plugins install <id>          # from the online registry, hash-verified
   picasso plugins update <id>
   picasso plugins uninstall <id>

``enable`` and ``install`` ask for confirmation, since both mean running
someone else's code; add ``--yes`` to skip the prompt in a script that has
already made that decision.

``picasso plugins list`` shows, for each enabled plugin, whether it adds a GUI
menu entry, which ``picasso`` commands it contributes and which API names it
exports:

.. code-block:: text

   Plugins in /home/you/.picasso/plugins:

     [x] my_plugin.py  (registry: my_plugin 1.2.0)
           GUI menu entry
           commands: count-locs
           API: count_locs
     [ ] draft.py  (local file)

To install a plugin on a machine with no GUI at all, either use
``picasso plugins install`` or copy the ``.py`` file into
``~/.picasso/plugins`` and run ``picasso plugins enable``.

If a plugin cannot be loaded, it is skipped and the rest of Picasso is
unaffected. Every ``picasso`` command then prints a one-line warning naming
the file; ``picasso plugins list`` shows the full traceback, and
``picasso plugins disable <file.py>`` stops it being loaded at all.

The enable flag is one shared setting: whether a plugin runs in a GUI, on the
command line or inside a script, it runs only after you have enabled it, and
disabling it stops all three. Enabling from the command line and from the
plugin browser is the same act, recorded in the same place.

.. _plugins-for-developers:

For developers
--------------
To create a plugin, you can use the template provided in
`picasso/plugin_template.py <https://github.com/jungmannlab/picasso/blob/master/plugin_template.py>`_,
which shows all three hooks; delete the ones you do not need. For more
examples of plugins, please see the
`GitHub repo <https://github.com/jungmannlab/picasso_plugins>`_.

A plugin file is read for three optional, independent things, and may define
any subset of them:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - What you define
     - What it extends
   * - a ``Plugin`` class
     - the Plugins menu of one Picasso GUI
   * - a ``PICASSO_API`` mapping
     - the names available to scripts as ``picasso.plugins.api.<name>``
   * - a ``register_cli`` function
     - the ``picasso`` command line

.. _plugins-extending-gui:

Extending a GUI
~~~~~~~~~~~~~~~
Define a ``Plugin`` class whose ``__init__(self, window)`` sets ``self.name``
to the app it targets (``"render"``, ``"localize"``, ``"filter"``,
``"average"``, ``"design"``, ``"simulate"``, ``"nanotron"`` or ``"spinna"``)
and stores ``window``. Its ``execute`` method is called when the app opens,
and adds actions to ``window.plugin_menu``:

.. code-block:: python

   class Plugin:
       def __init__(self, window):
           self.name = "render"
           self.window = window

       def execute(self):
           action = self.window.plugin_menu.addAction("Do the thing")
           action.triggered.connect(self.run)

       def run(self):
           ...

``execute`` is called again whenever the app rebuilds its menu bar, so it
should only add menu entries and not do work of its own.

.. _plugins-extending-api:

Extending the Python API
~~~~~~~~~~~~~~~~~~~~~~~~
Define ``PICASSO_API`` as a dict mapping the name a script will use to the
object you provide (a function, a class, a constant). It may also be a
function returning such a dict, if you want to build it lazily:

.. code-block:: python

   def count_locs(path):
       ...

   PICASSO_API = {"count_locs": count_locs}

Your users then reach it from any script or notebook:

.. code-block:: python

   from picasso import plugins

   print(plugins.api.count_locs("locs.hdf5"))

``plugins.api`` imports the enabled plugins the first time an attribute is
read; simply importing Picasso never runs plugin code. Use
``plugins.api.refresh()`` after enabling or editing a plugin in a running
session, or ``plugins.load_api_plugins()`` to get the whole mapping as a plain
dict.

If two enabled plugins export the same name, the first one keeps it and a
warning naming both files is printed, so adding a plugin can never silently
change what an existing script does.

.. _plugins-extending-cli:

Extending the command line
~~~~~~~~~~~~~~~~~~~~~~~~~~
Define ``register_cli(subparsers)``. It is called with the subparsers of the
``picasso`` argument parser; add one parser per subcommand and name its
handler with ``set_defaults(func=...)``. The handler is called with the parsed
``argparse.Namespace``:

.. code-block:: python

   def _run(args):
       for path in args.files:
           print(path, count_locs(path))

   def register_cli(subparsers):
       parser = subparsers.add_parser(
           "count-locs", help="count localizations in hdf5 files"
       )
       parser.add_argument("files", nargs="+")
       parser.set_defaults(func=_run)

The command then behaves like any built-in one:

.. code-block:: bash

   picasso count-locs *.hdf5

Command names may contain lowercase letters, digits, ``_`` and ``-``, and
cannot shadow a built-in Picasso command or one an earlier plugin already
registered - such a registration is refused and reported, leaving the rest of
the command line working.

.. important::

   Keep ``register_cli`` to building parsers: it runs on every ``picasso``
   invocation, so do the actual work (and any expensive import) inside the
   handler.

.. note::

   A plugin that only extends the API or the command line needs no ``Plugin``
   class and no PyQt at all. It still lives in the same folder and obeys the
   same enable gate.

.. _plugins-documenting:

Documenting your plugin
~~~~~~~~~~~~~~~~~~~~~~~
**Document your plugin in the module docstring at the top of the file.**
Picasso has no separate place to show a plugin's documentation, and users read
plugins through **View source**, so the docstring that opens the ``.py`` file
is the plugin's manual.

- Put the usage instructions there - what the plugin does, which menu entry it
  adds, what input it expects and what it produces, plus any requirements or
  caveats.
- Place it before the imports and any other code, so it is the first thing
  the user sees.
- Keep further explanation of the implementation in the docstrings of the
  individual functions and classes.

.. _plugins-registry-entries:

Registry entries
~~~~~~~~~~~~~~~~
Every entry in the registry's ``index.json`` must carry the SHA-256 of the
file it points to, or Picasso will not install it:

.. code-block:: json

   {
     "id": "render_extras",
     "display_name": "Render extras",
     "app": "render",
     "description": "Extra export actions for Render.",
     "author": "Jungmann Lab",
     "version": "1.0.0",
     "file": "plugins/render_extras.py",
     "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
     "min_picasso_version": "0.11.0"
   }

``id``
   Must consist of letters, digits, ``_`` and ``-`` only (it becomes the local
   file name).
``file``
   Must be a relative path to a ``.py`` file inside the registry repository.
``app``
   The Picasso app the plugin is listed under; use ``"api"`` for a plugin that
   extends the Python API or the command line rather than one GUI, and it will
   be listed in every app's plugin browser.

Entries that violate either rule (``id`` or ``file``) are ignored. Whenever
you change a plugin file, update both ``version`` and ``sha256`` in the same
commit - the manifest is what pins the code, so the hash is what makes the
change reviewable.

Note that plugin state (which files are installed and which are enabled) lives
in a hidden ``.installed.json`` sidecar in the plugins folder.
