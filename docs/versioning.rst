.. _versioning:

Stability and versioning
========================

Since version 1.0, Picasso follows `semantic versioning <https://semver.org>`_
for what this page lists as public: scripts, pipelines, plugins and files that
work with Picasso 1.0 keep working with every 1.x release. What changes, and
when, follows from the version number ``MAJOR.MINOR.PATCH``:

- A **patch** release (``1.0.1``) fixes bugs only.
- A **minor** release (``1.1``) adds features and may deprecate public names;
  it removes only names whose deprecation period is over (see
  :ref:`versioning-deprecations`).
- A **major** release (``2.0``) may remove or change anything public, e.g.,
  the meaning of a column of the localization files.

.. _versioning-public:

What is public
--------------

**The Python API**
   The functions and classes on the :doc:`api/index` pages, which are the names
   in each module's ``__all__``, with their parameters, defaults, return values
   and units. Everything else may change in any release: names starting with
   an underscore, names left out of ``__all__``, and the modules that are not
   on the API pages, e.g., ``picasso.gui``, ``picasso.lib_qt``,
   ``picasso.server`` and the submodules of ``picasso.render``.

**The files**
   Every Picasso 1.x reads the localization files written by any earlier
   version of Picasso, including 0.x, and an older 1.x opens the files of a
   newer 1.x, ignoring what it does not know. A new major version of the file
   format, which older versions refuse, comes only with a major release of
   Picasso; see :ref:`files-format-versions`. The format version is
   ``picasso.io.FORMAT_VERSION`` (1.0 since Picasso 1.0: every length in nm;
   files from before 1.0, in camera pixels, are converted when opened, see
   :ref:`files-legacy`). The same holds for the other files Picasso writes,
   e.g., 3D and spline calibrations, picks and SPINNA structures. The metadata keys documented in :doc:`files` are public; any
   other key is informational. The user settings file is not: unknown keys
   are ignored and missing ones take their defaults.

**The command line**
   The commands, their options and the names of the files they write (e.g.,
   ``<file name>_undrift.hdf5``), see :doc:`cmd`.

**Plugins**
   The three hooks a plugin uses: a ``Plugin`` class added to a window's
   ``plugin_menu``, ``PICASSO_API`` and ``register_cli``, see
   :ref:`plugins-for-developers`. Plugins may use the public Python API. The
   other attributes of the Picasso windows (e.g., ``window.view``) are not
   public, so check GUI plugins against each minor release.

Not public: the windows themselves (menus, dialogs, layout, keyboard
shortcuts), the text printed to the console.

What is a breaking change
-------------------------

Breaking, so only after a deprecation period or in a major release:

- removing or renaming a public name, parameter, command or option;
- changing the order of positional parameters;
- changing a return type, a unit or the meaning of a value (e.g., how dark
  times are counted);
- changing a default so that the results change.

Not breaking, so possible in any minor release:

- new functions, classes, commands and options;
- new optional (keyword) parameters;
- new columns in returned tables and saved files;
- bug fixes, also those that change wrong results. The changelog lists them.

.. _versioning-deprecations:

Deprecations
------------

Names deprecated before Picasso 1.0 are removed in 1.0 (see :doc:`migration`
for their replacements). From then on, a public
name deprecated in version 1.x keeps working until 1.(x+2) at the earliest, i.e.,
for at least two minor releases: a name deprecated in 1.1
still works in 1.1 and 1.2 and may be removed in 1.3. Using it shows a
``FutureWarning`` that names the replacement and the version that removes the
name, and its API page has a *Deprecated* note. The changelog lists the
deprecations and removals of each release.

The commands and options renamed in Picasso 1.0 keep their old names until
Picasso 2.0, see :ref:`cmd-renamed`.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``FutureWarning`` is shown by default. To find every deprecated name a
   script uses, turn the warnings into errors:

   .. code-block:: python

      import warnings

      warnings.simplefilter("error", FutureWarning)

.. _versioning-python:

Supported Python versions
-------------------------

Picasso follows `SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_,
the schedule of NumPy, SciPy, scikit-learn and other scientific Python
projects:

- each Python version is supported for at least **three years** after its
  release;
- each version of the core libraries Picasso builds on (NumPy, SciPy,
  Matplotlib, pandas, scikit-learn) is supported for at least **two years**
  after its release.

Picasso may support a version longer, e.g., while a newer one is not yet usable
with all of Picasso's dependencies. A version past its window may be dropped in
a minor release; the changelog announces it, and the earlier releases of
Picasso stay installable for it. The versions supported now are listed in
``pyproject.toml`` (Python 3.11 to 3.14 in Picasso 1.0).

.. _versioning-developers:

For developers
--------------

Deprecate with ``picasso.deprecation``. Its ``deprecated`` decorator warns,
adds the note to the docstring and records the versions:

.. code-block:: python

   from picasso.deprecation import deprecated

   @deprecated(since="1.1", remove_in="1.3", alternative="picasso.lib.new_name")
   def old_name(locs, info):
       return new_name(locs, info)

For anything else (a parameter, a module), register the deprecation at module
level and warn where the old name is used:

.. code-block:: python

   from picasso import deprecation

   _OLD_PARAMETER = deprecation.register(
       "picasso.postprocess.link(max_dark)", since="1.1", remove_in="1.3",
       alternative="max_dark_time",
   )

   def link(locs, info, max_dark_time=1, max_dark=None):
       if max_dark is not None:
           _OLD_PARAMETER.warn()
           max_dark_time = max_dark
       ...

``tests/test_stability.py`` keeps the policy:

- it compares the public names (``__all__`` of the modules on the API pages)
  and the commands and options with snapshots in ``tests/data/stability``, so
  a change of them fails until it is recorded with
  ``python tests/regen_stability_snapshots.py``. The script refuses to
  record the removal of a public name (a rename removes the old name) unless
  the name is deprecated and this version may remove it (``remove_in``, its
  pre-releases, or later). ``--allow-breaking`` records any removal, for a
  major release or a name that was public by mistake;
- it fails once a registered deprecation is due for removal (from the final
  release ``remove_in``; its pre-releases may still contain the name), and
  ``register`` refuses a removal sooner than the policy allows.

Before each release:

1. Update the version in ``picasso/version.py``.
2. Clean up the changelog section of this version in ``changelog.md`` and
   update its "Last change" date. The section's heading must be the new
   version, or ``release/make_release_notes.py`` refuses to write the
   release notes.
3. Run the tests locally (``python -m pytest``).
4. Remove the deprecated names due in this version (the test lists them) and
   list the removals in the changelog.
5. If the file formats changed, bump ``io.FORMAT_VERSION`` (minor for
   additions, major for changes of meaning; 1.0 is the first versioned format,
   with every length in nm) and add a row to the table in
   :ref:`files-format-versions`.
6. Build the one-click installers with the scripts in ``release`` on Windows
   and macOS. Each script tests the frozen app (``release/smoke_test.py``)
   before packaging it and stops if the test fails, and zips the installer
   with its ``readme.txt``. The standard and the CUDA Windows installers can
   be built at the same time.
7. Publish the release on GitHub, with a tag ``v`` plus the version in
   ``picasso/version.py`` (e.g., ``v1.1.0``) and the three zips attached.
   Keep their names as the scripts write them: *Install Now* in the update
   notification finds the installer of each edition by name ("Windows",
   "CUDA" for the CUDA edition, and "Apple" or "macOS"), see
   ``picasso.updater.find_installer``.
   ``python release/make_release_notes.py`` writes its description, from the
   changelog section of this version, to ``release/release_notes.txt``, to
   paste into the release.
   ``.github/workflows/pypi_release.yml`` then builds the package automatically,
   tests the wheel on Linux, macOS and Windows (``release/smoke_test.py``) and
   publishes it on PyPI. To try a release candidate first, run the workflow
   by hand, which publishes on TestPyPI.
