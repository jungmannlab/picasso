Picasso
=======
.. image:: https://readthedocs.org/projects/picassosr/badge/?version=latest
   :target: https://picassosr.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/jungmannlab/picasso/workflows/CI/badge.svg
   :target: https://github.com/jungmannlab/picasso/workflows/CI/badge.svg
   :alt: CI

.. image:: http://img.shields.io/badge/DOI-10.1038/nprot.2017.024-52c92e.svg
   :target: https://doi.org/10.1038/nprot.2017.024
   :alt: DOI

.. image:: https://static.pepy.tech/personalized-badge/picassosr?period=total&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads
   :target: https://pepy.tech/project/picassosr
   :alt: Downloads

.. image:: https://img.shields.io/pypi/pyversions/picassosr
   :target: https://pypi.org/project/picassosr/
   :alt: Python versions

.. image:: https://img.shields.io/pypi/v/picassosr
   :target: https://pypi.org/project/picassosr/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/Changelog-View-blue
   :target: https://github.com/jungmannlab/picasso/blob/master/changelog.md
   :alt: Changelog

.. image:: https://raw.githubusercontent.com/jungmannlab/picasso/master/main_render.png
   :width: 750
   :height: 564
   :alt: UML Render view

Collection of tools for painting super-resolution images. The Picasso software is complemented by our `Nature Protocols publication <https://www.nature.com/nprot/journal/v12/n6/abs/nprot.2017.024.html>`__.

A comprehensive documentation can be found here: `Read the Docs <https://picassosr.readthedocs.io/en/latest/?badge=latest>`__.

To see all changes introduced across releases, see `the changelog <https://github.com/jungmannlab/picasso/blob/master/changelog.md>`_.

Picasso 0.11
------------
This release substantially expands Picasso: Localize. Localization can now be performed with an experimentally measured PSF (cubic-spline model), jointly across several channels (e.g. biplane 3D), and with a pixel-dependent sCMOS noise model; rotated and spherical 2D Gaussian models were added as well. All GPU fitting was reimplemented in Numba CUDA, removing the dependency on Gpufit. Localize also reads a much wider range of data directly - ``.tif`` and OME-TIFF stacks (including movies split across several folders), MicroManager single-image acquisitions, Zeiss ``.czi`` and Leica ``.lif`` - so Picasso: ToRaw is no longer required and has been removed. Further additions include a temporal median filter for spot identification, affine calibrations for astigmatism and chromatic aberration correction, localization metadata embedded in the ``.hdf5`` files, a revised plugin system with an online plugin browser, and various performance and usability improvements throughout Localize, Render and SPINNA. We encourage all users to acquaint themselves with the new features in the `Localize documentation <https://picassosr.readthedocs.io/en/latest/localize.html>`_. See the `changelog <https://github.com/jungmannlab/picasso/blob/master/changelog.md>`_ for the complete list.

Installation
------------

Check out the `Picasso release page <https://github.com/jungmannlab/picasso/releases/>`__ to download and run the latest compiled one-click installer for Windows or MacOS (the latter is experimental and feedback is welcome). Here you will also find the Nature Protocols legacy version (v0.1.0).

For Windows, two one-click installers are provided: a default build and a **CUDA** build. Both render on the graphics card in Picasso: Render (via ``wgpu``, any vendor). The CUDA build additionally bundles the CUDA runtime so that CUDA-accelerated (numba.cuda) code can run, for example localization fitting. It is larger and requires an NVIDIA (CUDA-capable) GPU; on machines without one, CUDA-only options are simply hidden. Choose the CUDA installer only if you have a compatible NVIDIA GPU and want to use the accelerated fitting tools. Picasso uses CUDA 12 in the one-click installer.

Python is also distributed as a PyPI package that is platform-independent (``pip install picassosr``) which grants not only GUI but also access to Picasso’s internal routines in custom Python programs. For more details, see the `Via PyPI <https://github.com/jungmannlab/picasso#via-pypi>`__ section below. For examples of how to use Picasso in Python scripts, see the section `Example Usage <https://github.com/jungmannlab/picasso#example-usage>`__ below.


Via PyPI
^^^^^^^^

1. Open the console/terminal and create a new conda environment: ``conda create --name picasso python=3.14``. Note you can use other Python versions as well.
2. Activate the environment: ``conda activate picasso``.
3. Install Picasso package using: ``pip install picassosr``.
4. You can now run any Picasso function directly from the console/terminal by running: ``picasso render``, ``picasso localize``, etc, or import Picasso functions in your own Python scripts.
5. To update Picasso (you should get a notification about available updates since v0.10.0) run ``pip install --upgrade picassosr``.
6. You can optionally install dependencies for .czi and .lif formats by passing ``pip install picassosr[czi]`` or ``pip install picassosr[lif]``.
7. To enable GPU-accelerated (numba.cuda) code, install the CUDA dependencies with ``pip install picassosr[gpu]``. This requires an NVIDIA (CUDA-capable) GPU. The ``gpu`` extra targets CUDA toolkit 12.x; for other toolkits use ``pip install picassosr[cuda11]`` or ``pip install picassosr[cuda13]`` instead. Without these extras, Picasso runs fine on the CPU and GPU-only options are hidden.

For Developers (local, editable installation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to use your local version of Picasso with your own modifications:

1. Open the console/terminal and create a new conda environment: ``conda create --name picasso python=3.14``. Note you can use other Python versions as well.
2. Activate the environment: ``conda activate picasso``.
3. Change to the directory of choice using ``cd``.
4. Clone this GitHub repository by running ``git clone https://github.com/jungmannlab/picasso``. Alternatively, `download <https://github.com/jungmannlab/picasso/archive/master.zip>`__ the zip file and unzip it.
5. Open the Picasso directory: ``cd picasso``.
6. You can modify Picasso code in this directory.
7. To create a *local* Picasso package to use it in other Python scripts, run ``pip install -e ".[dev]"``. When you change the code in the ``picasso`` directory, the changes will be reflected in the package.
8. You can install other extensions, such as ``".[gpu]"``, etc. The whole list of optional dependencies can be found in ``pyproject.toml``.
9. You can now run any Picasso module directly from the console/terminal by running: ``picasso render``, ``picasso localize``, etc, or import Picasso functions in your own Python scripts.

Creating shortcuts on Windows (*optional*)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This applies only to the users who installed Picasso via PyPI or through the editable, developer version and want to use desktop shortcuts. If you installed Picasso from the one-click installer on the `Release page <https://github.com/jungmannlab/picasso/releases/>`__, you can ignore this section. Run the PowerShell script “createShortcuts.ps1” in the ``gui`` directory. This should be doable by right-clicking on the script and choosing “Run with PowerShell”. Alternatively, run the command
``powershell ./createShortcuts.ps1`` in the command line. Use the generated shortcuts in the top level directory to start GUI components. Users can drag these shortcuts to their Desktop, Start Menu or Task Bar.

Example Usage
-------------

Besides using the GUI, you can use picasso like any other Python module. Consider the following example:::

  from picasso import io, postprocess

  path = 'testdata_locs.hdf5'
  locs, info = io.load_locs(path)
  
  # Link localizations and calculate dark times
  linked_locs = postprocess.link(picked_locs, info, r_max=0.05, max_dark_time=1)
  linked_locs_dark = postprocess.compute_dark_times(linked_locs)

  print(f"Average bright time {linked_locs_dark['n'].mean():.2f} frames")
  print(f"Average dark time {linked_locs_dark['dark'].mean():.2f} frames")

For more examples, visit the `sample notebooks <https://github.com/jungmannlab/picasso/tree/master/samples>`__.

Contributing
------------

If you have a feature request or a bug report, please post it as an issue on the GitHub issue tracker. If you want to contribute, put a pull request (PR) for it. You can find more guidelines for contributing `here <https://github.com/jungmannlab/picasso/blob/master/CONTRIBUTING.rst>`__. We will gladly guide you through the codebase and credit you accordingly.  You can also contact us via picasso@jungmannlab.org.

.. SYNC-START: contributions

Contributions & Copyright
-------------------------

| Contributors: Joerg Schnitzbauer, Maximilian Strauss, Rafal Kowalewski, Adrian Przybylski, Andrey Aristov, Hiroshi Sasaki, Alexander Auer, Johanna Rahm
| Copyright (c) 2015-2025 Jungmann Lab, Max Planck Institute of Biochemistry

.. SYNC-END: contributions

.. SYNC-START: citing

Citing Picasso
--------------

If you use Picasso in your research, please cite our Nature Protocols publication describing the software.

| J. Schnitzbauer*, M.T. Strauss*, T. Schlichthaerle, F. Schueder, R. Jungmann
| Super-Resolution Microscopy with DNA-PAINT
| Nature Protocols (2017). 12: 1198-1228 DOI: `10.1038/nprot.2017.024 <https://doi.org/10.1038/nprot.2017.024>`__
|
| If you use some of the functionalities provided by Picasso, please also cite the respective publications:

- All fitting methods are ports of Gpufit. DOI: `10.1038/s41598-017-15313-9 <https://doi.org/10.1038/s41598-017-15313-9>`__. License can be found `here <https://github.com/jungmannlab/picasso/blob/master/LICENSES/Gpufit-LICENSE.txt>`__.
- Experimental PSF (cubic-spline) fitting. DOIs: `10.1038/nmeth.4661 <https://doi.org/10.1038/nmeth.4661>`__ (Li et al., experimental-PSF localization and bead alignment) and `10.1038/s41598-017-00622-w <https://doi.org/10.1038/s41598-017-00622-w>`__ (Babcock & Zhuang, cubic-spline PSF model). The spline calibration follows the coefficient scheme of Gpuspline; license can be found `here <https://github.com/jungmannlab/picasso/blob/master/LICENSES/Gpuspline-LICENSE.txt>`__.
- Multichannel (global) experimental-PSF fitting. DOI: `10.1038/s41467-022-30719-4 <https://doi.org/10.1038/s41467-022-30719-4>`__ (Li et al., globLoc).
- 3D fitting via astigmatism. DOI: `10.1126/science.1153529 <https://www.science.org/doi/10.1126/science.1153529>`__.
- B-spline wavelet spot identification. DOI: `10.1364/OE.20.002081 <https://doi.org/10.1364/OE.20.002081>`__ (Izeddin et al., Opt. Express 2012)
- sCMOS pixel-dependent noise modeling. DOI: `10.1038/nmeth.2488 <https://doi.org/10.1038/nmeth.2488>`__.
- NeNA. DOI: `10.1007/s00418-014-1192-3 <https://doi.org/10.1007/s00418-014-1192-3>`__
- FRC. DOI: `10.1038/nmeth.2448 <https://doi.org/10.1038/nmeth.2448>`__
- Theoretical lateral localization precision (``lpx`` / ``lpy``, Gaussian least-squares). DOI: `10.1038/nmeth.1447 <https://doi.org/10.1038/nmeth.1447>`__
- Theoretical axial localization precision (``lpz`` values, Gaussian). DOI: `10.1038/s41467-026-70198-5 <https://doi.org/10.1038/s41467-026-70198-5>`__
- Quad-tree adaptive histogram rendering. DOI: `10.1017/S143192760999122X <https://doi.org/10.1017/S143192760999122X>`__ (Baddeley, Cannell & Soeller, Microsc. Microanal. 2010)
- RCC undrifting: DOI: `10.1364/OE.22.015982 <https://doi.org/10.1364/OE.22.015982>`__
- AIM undrifting. DOI: `10.1126/sciadv.adm776 <https://www.science.org/doi/10.1126/sciadv.adm7765>`__
- SMLM clusterer. DOIs: `10.1038/s41467-021-22606-1 <https://doi.org/10.1038/s41467-021-22606-1>`__ and `10.1038/s41586-023-05925-9 <https://doi.org/10.1038/s41586-023-05925-9>`__
- DBSCAN: Ester, et al. Inkdd, 1996. (Vol. 96, No. 34, pp. 226-231).
- Anisotropic DBSCAN inspired by: `10.1021/acs.jpcb.4c02030 <https://doi.org/10.1021/acs.jpcb.4c02030>`__
- HDBSCAN. DOI: `10.1007/978-3-642-37456-2_14 <https://doi.org/10.1007/978-3-642-37456-2_14>`__
- RESI. DOI: `10.1038/s41586-023-05925-9 <https://doi.org/10.1038/s41586-023-05925-9>`__
- Nanotron. DOI: `10.1093/bioinformatics/btaa154 <https://doi.org/10.1093/bioinformatics/btaa154>`__
- Picasso: Server. DOI: `10.1038/s42003-022-03909-5 <https://doi.org/10.1038/s42003-022-03909-5>`__
- SPINNA. DOI: `10.1038/s41467-025-59500-z <https://doi.org/10.1038/s41467-025-59500-z>`__
- SPINNA for LE fitting. DOI: `10.1038/s41592-024-02242-5 <https://doi.org/10.1038/s41592-024-02242-5>`__
- G5M. DOI: `10.1038/s41467-026-70198-5 <https://doi.org/10.1038/s41467-026-70198-5>`__

.. SYNC-END: citing

.. SYNC-START: credits

Credits
-------

-  Design icon based on “Hexagon by Creative Stalls" from the Noun Project
-  Simulate icon based on “Microchip by Futishia" from the Noun Project
-  Localize icon based on “Mountains" by MONTANA RUCOBO from the Noun Project
-  Filter icon based on “Funnel" by José Campos from the Noun Project
-  Render icon based on “Paint Palette" by Vectors Market from the Noun Project
-  Average icon based on “Layers" by Creative Stall from the Noun Project
-  Server icon based on “Database" by Nimal Raj from the Noun Project
-  SPINNA icon based on "Spinner" by Viktor Ostrovsky from the Noun Project

.. SYNC-END: credits
