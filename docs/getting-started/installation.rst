Installation
============

Picasso runs on Windows, macOS and Linux. Pick the route that fits you:

- **One-click installer** if you only want to use the Picasso windows.
- **PyPI** if you also want to use Picasso in your own Python scripts.
- **Developer installation** if you want to change Picasso's code.

.. tab-set::

   .. tab-item:: One-click installer
      :sync: installer

      Download the latest installer for Windows or macOS from the `Picasso
      release page <https://github.com/jungmannlab/picasso/releases/>`__ and
      run it. The macOS installer is experimental, and feedback is welcome.
      The release page also hosts the Nature Protocols legacy version
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
      and want the accelerated fitting tools. On machines without one,
      CUDA-only options are hidden.

   .. tab-item:: PyPI
      :sync: pypi

      Picasso is distributed as the platform-independent PyPI package
      ``picassosr``. It provides the GUI and access to Picasso's routines in
      your own Python programs.

      1. Create and activate a new conda environment (other Python versions
         work as well):

         .. code-block:: bash

            conda create --name picasso python=3.14
            conda activate picasso

      2. Install Picasso:

         .. code-block:: bash

            pip install picassosr

      3. Start any module from the terminal, for example ``picasso render``
         or ``picasso localize``, or import Picasso in your scripts.

      .. rubric:: Optional extras

      .. list-table::
         :header-rows: 1
         :widths: 35 65

         * - Command
           - Adds
         * - ``pip install picassosr[czi]``
           - Reading Zeiss ``.czi`` files.
         * - ``pip install picassosr[lif]``
           - Reading Leica ``.lif`` files.
         * - ``pip install picassosr[gpu]``
           - GPU-accelerated (``numba.cuda``) code for CUDA toolkit 12.x.
             Needs an NVIDIA (CUDA-capable) GPU.
         * - ``pip install picassosr[cuda11]`` / ``[cuda13]``
           - The same for CUDA toolkit 11.x or 13.x.

      Without the GPU extras, Picasso runs on the CPU and GPU-only options
      are hidden.

      .. rubric:: Updating

      Picasso notifies you about new versions (since v0.10.0). To update,
      run:

      .. code-block:: bash

         pip install --upgrade picassosr

   .. tab-item:: Developer installation
      :sync: dev

      Use a local, editable installation to work with your own changes to
      Picasso:

      1. Create and activate a new conda environment:

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
         full list is in ``pyproject.toml``.

      4. Start any module from the terminal, for example ``picasso render``,
         or import Picasso in your scripts.

      See :doc:`/development` for how to contribute your changes.

Desktop shortcuts on Windows
----------------------------

The one-click installer creates shortcuts for you. For a PyPI or developer
installation, run the PowerShell script ``createShortcuts.ps1`` in the
``picasso/gui`` directory, either by right-clicking it and choosing *Run with
PowerShell*, or with:

.. code-block:: bash

   powershell ./createShortcuts.ps1

Use the generated shortcuts in the top-level directory to start the modules.
You can drag them to the Desktop, Start menu or taskbar.

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
