Installation
============

Picasso runs on Windows, macOS and Linux. Several installation modes are available:

- **One-click installer** if you only want to use the Picasso GUI.
- **PyPI** if you also want to use Picasso in your own Python scripts.
- **Developer installation** if you want to change Picasso's code.

.. tab-set::

   .. tab-item:: One-click installer
      :sync: installer

      Download the latest installer for Windows or macOS from the `Picasso
      release page <https://github.com/jungmannlab/picasso/releases/>`__ and
      run it. The release page also hosts the Nature Protocols legacy version
      (v0.1.0).

      .. rubric:: Windows: default or CUDA build?

      There are two Windows installers:

      - The **default** build.
      - The **CUDA** build, which also bundles the CUDA runtime (CUDA 12)
        so that CUDA-accelerated (``numba.cuda``) code runs, for example
        localization fitting on the GPU. It is larger and needs an NVIDIA
        (CUDA-capable) GPU.

      Both builds render on the graphics card in Render (via ``wgpu``, any
      vendor). Choose the CUDA build only if you have a compatible NVIDIA GPU
      and want the accelerated tools. On machines without one, CUDA-only
      options are hidden.

   .. tab-item:: PyPI
      :sync: pypi

      Picasso is distributed as the platform-independent PyPI package
      ``picassosr``. It provides the GUI and access to Picasso's routines in
      your own Python programs.

      1. Create and activate a new conda environment. Picasso supports Python
         3.11 to 3.14:

         .. code-block:: bash

            conda create --name picasso python=3.14
            conda activate picasso

         .. note::

            Python 3.15 is not supported yet: some of Picasso's dependencies
            cannot be installed on it with pip, and GPU fitting (CUDA) would
            not work. See :ref:`faq-python-315`.

      2. Install Picasso:

         .. code-block:: bash

            pip install picassosr

      3. Start any module from the terminal, for example ``picasso render``
         or ``picasso localize``, or import Picasso in your scripts.

      .. rubric:: Menu entries on Linux

      The script ``create_linux_shortcuts.py`` adds Picasso's modules to the
      application menu of a Linux desktop (GNOME, KDE, Xfce, ...), and adds
      a ``picasso`` command that works without activating the environment.
      Download it and run it with the Python of the environment Picasso is
      installed in:

      .. code-block:: bash

         conda activate picasso
         curl -O https://raw.githubusercontent.com/jungmannlab/picasso/master/release/create_linux_shortcuts.py
         python create_linux_shortcuts.py

      The menu entries are written to ``~/.local/share/applications`` and
      the command to ``~/.local/bin``. Run as root, with
      ``sudo "$(which python)" create_linux_shortcuts.py``, the script adds
      them for all users (to ``/usr/share/applications`` and
      ``/usr/local/bin``). Run it again after installing Picasso in another
      environment, and ``python create_linux_shortcuts.py --remove`` to
      delete the entries. The Server entry is added with the ``server``
      extra installed.

      .. rubric:: Optional extras

      .. list-table::
         :header-rows: 1
         :widths: 35 65

         * - Command
           - Adds
         * - ``pip install picassosr[server]``
           - :doc:`Picasso Server <../server>` (``picasso server``), the
             browser app.
         * - ``pip install picassosr[czi]``
           - Reading Zeiss ``.czi`` files.
         * - ``pip install picassosr[lif]``
           - Reading Leica ``.lif`` files.
         * - ``pip install picassosr[gpu]``
           - GPU-accelerated (``numba.cuda``) code for CUDA toolkit 12.x.
             Needs an NVIDIA (CUDA-capable) GPU.
         * - ``pip install picassosr[cuda11]`` / ``[cuda13]``
           - The same for CUDA toolkit 11.x or 13.x.

      :ref:`GPU rendering <render-gpu-rendering>` in Render (``wgpu``: Metal
      on macOS, Direct3D 12 on Windows, Vulkan on Linux; any recent graphics
      card) is included in the default installation. Without the CUDA
      extras, CUDA-only options are hidden.

      .. rubric:: Updating

      Picasso notifies you about new versions (since v0.10.0), at most once
      a day. In the command line, the notification comes after the command
      has finished, and Picasso asks what to do about it only in a
      terminal, so scripts and batch jobs never wait for an answer. To
      switch the checks off, e.g., on a computing cluster, set ``Disabled``
      in the :ref:`user settings <user-settings-file>`. To update, run:

      .. code-block:: bash

         pip install --upgrade picassosr

   .. tab-item:: Developer installation
      :sync: dev

      Use a local, editable installation to work with your own changes to
      Picasso:

      1. Create and activate a new conda environment with Python 3.11 to
         3.14 (not 3.15 yet, see :ref:`faq-python-315`):

         .. code-block:: bash

            conda create --name picasso python=3.14
            conda activate picasso

      2. Clone the repository (or `download the zip file
         <https://github.com/jungmannlab/picasso/archive/master.zip>`__ and
         unzip it) and change into it:

         .. code-block:: bash

            git clone https://github.com/jungmannlab/picasso
            cd picasso

      3. Install Picasso in editable mode. Changes to the code in the
         ``picasso`` directory take effect without reinstalling:

         .. code-block:: bash

            pip install -e ".[dev]"

         Other extras, such as ``".[gpu]"``, can be added the same way; the
         full list is in `pyproject.toml
         <https://github.com/jungmannlab/picasso/blob/master/pyproject.toml>`__.

      4. Start any module from the terminal, for example ``picasso render``,
         or import Picasso in your scripts.

      .. tip::

         On Windows, you can create shortcuts for the modules by running
         this from the repository with the environment activated:

         .. code-block:: bash

            powershell -ExecutionPolicy Bypass -File picasso\gui\createShortcuts.ps1

         The shortcuts appear in the repository folder; drag them to the
         Desktop, Start menu or taskbar.

      See :doc:`/development` for how to contribute your changes.

.. _updating-to-1-0:

Updating from an earlier version
--------------------------------

**Your files keep working.** Since Picasso 1.0, every length (coordinates,
precisions, pick sizes, drift, ...) is stored in nm instead of camera pixels.
Picasso 1.0 reads all files written by earlier versions as before and converts
them to nm as it opens them, without changing them on disk; there is nothing
you need to do. See :ref:`files-legacy` for details, including how to convert
files on disk if you want to.

What has changed is how lengths are entered: dialogs, command line options and
Python functions now take nm. Scripts that pass camera pixels need updating;
:doc:`/migration` lists what changes and the replacement of each setting,
command and function.

.. _error-log:

Reporting problems
------------------

When an error occurs, Picasso shows it in a message box (click
*Show Details...* for the full traceback) and appends the traceback to the
log file ``~/.picasso/logs/picasso.log`` (i.e.,
``C:\Users\<you>\.picasso\logs\picasso.log`` on Windows), together with
the tracebacks of failing background threads. The file rotates to
``picasso.log.1`` once it exceeds 5 MB. The one-click installers have no
console, so everything Picasso prints goes to this file as well.

Many problems are answered in the :doc:`/faq`. Please report others on
`GitHub <https://github.com/jungmannlab/picasso/issues>`__ and attach the log
file.

Next steps
----------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket;1.5em;sd-mr-1` First steps
      :link: workflow
      :link-type: doc
      :class-card: sd-card-hover

      Follow a DNA-PAINT analysis from raw movie to final image.

   .. grid-item-card:: :octicon:`code;1.5em;sd-mr-1` Use Picasso in Python
      :link: /api/index
      :link-type: doc
      :class-card: sd-card-hover

      Load localizations, postprocess and render them in your own scripts.
