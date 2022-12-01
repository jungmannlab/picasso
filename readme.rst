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
   :alt: CI

.. image:: https://static.pepy.tech/personalized-badge/picassosr?period=total&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads
 :target: https://pepy.tech/project/picassosr

.. image:: main_render.png
   :scale: 100 %
   :alt: UML Render view

A collection of tools for painting super-resolution images. The Picasso software is complemented by our `Nature Protocols publication <https://www.nature.com/nprot/journal/v12/n6/abs/nprot.2017.024.html>`__.
A comprehensive documentation can be found here: `Read the Docs <https://picassosr.readthedocs.io/en/latest/?badge=latest>`__.

Picasso 0.5.0
-------------
Picasso has introduced many changes, including 3D rotation window and a new clustering algorithm in Render and reading of .nd2 files in Localize. Please check the `changelog <https://github.com/jungmannlab/picasso/blob/master/changelog.rst>`_ to see all modifications.

Picasso 0.4.0
-------------
Picasso now has a server-based workflow management-system. Check out `here <https://picassosr.readthedocs.io/en/latest/server.html>`__.


Installation
------------

Check out the `Picasso release page <https://github.com/jungmannlab/picasso/releases/>`__ to download and run the latest compiled one-click installer for Windows. Here you will also find the Nature Protocols legacy version. For the platform-independent usage of Picasso (e.g., with Linux and Mac Os X), please follow the advanced installation instructions below.

Advanced installation for Python programmers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an alternative to the stand-alone program for end-users, Picasso can be installed as a Python package. This is the preferred option to use Picasso’s internal routines in custom Python programs. For windows, one is still possible to use Picasso as an end-user by creating the respective shortcuts. This allows Picasso to be used on the same system by both programmers and end-users.

Requirements
^^^^^^^^^^^^

Python 3.8 (Tested on Windows 10)
'''''''''''''''''''''''''''''''''

We highly recommend the `Anaconda or Miniconda <https://www.continuum.io/downloads>`__ Python distribution which comes with a powerful package manager.

Setting up the environment with conda
'''''''''''''''''''''''''''''''''''''

Sample instructions to create an environment and installation of packages with conda are as follows:

1. Open the console and create a new conda environment: ``conda create --name picasso python=3.8``
2. Activate the environment: ``source activate picasso`` for Linux / Mac Os X or ``activate picasso`` for Windows.
3. (Optional) For Mac systems (e.g. M1) install PyQt via conda: ``conda install -c anaconda pyqt``.
4. Install using pip: ``pip install picassosr``.
5. (Optional) If you want to use hdbscan install using pip: ``pip install hdbscan``.
6. (Optional) If you plan to compile your own installer additionally install Pyinstaller: ``pip install pyinstaller``
7. Continue with the installation of Picasso (see the **Instalation (continued)** tab below)

Troubleshooting: In case installing via ``pip`` fails, try to install the failing packages via conda.

Note that sometimes outdated packages can cause problems. As of version 0.3.0, Picasso switched from PyQt4 to PyQt5, so make sure to update PyQt. If you experience errors, please check whether your packages have the correct version (e.g., see issue #4). When using conda, make sure that you have the default package channel (e.g., see issue #30).

.. _installation-1:

Installation (continued)
^^^^^^^^^^^^^^^^^^^^^^^^

There are two approaches to installing Picasso. Firstly, ``pip`` can be used to download Picasso release from `PyPI <https://pypi.org/project/picassosr/>`_. Alternatively, the GitHub repo can be directly cloned to your computer. Please see the instructions below for details.

Via PyPI:
'''''''''

1. Install using pip: ``pip install picassosr``.
2. Launch via calling one of the modules, e.g. ``picasso localize``.

Via GitHub:
'''''''''''

1. Open the console, ``cd`` to the directory where you want to install and run ``git clone https://github.com/jungmannlab/picasso``. Alternatively, `download <https://github.com/jungmannlab/picasso/archive/master.zip>`__ the zip file and unzip it.
2. Change to the downloaded directory ``cd picasso``
3. Run installation ``python setup.py install``.
4. Launch via calling one of the modules, e.g. ``picasso localize``.

Updating
^^^^^^^^
If Picasso was installed from PyPI, run the following command:
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

``pip install --upgrade picassosr``

If Picasso was cloned from the GitHub repo, use the following commands:
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

1. Move to the Picasso folder with the terminal, activate environment.
2. Update with git: ``git pull``.
3. Update the environment: ``pip install --upgrade -r requirements.txt``.
4. Run installation ``python setup.py install``.

Creating shortcuts on Windows (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the PowerShell script “createShortcuts.ps1” in the gui directory. This should be doable by right-clicking on the script and choosing “Run with PowerShell”. Alternatively, run the command
``powershell ./createShortcuts.ps1`` in the command line. Use the generated shortcuts in the top level directory to start GUI components. Users can drag these shortcuts to their Desktop, Start Menu or Task Bar.

Using Picasso as a module
^^^^^^^^^^^^^^^^^^^^^^^^^

The individual modules of picasso can be started as follows:
1. Open the console, activate the environment: ``source activate picasso`` for Linux / Mac Os X or ``activate picasso`` for Windows.
2. Start the picasso modules via ``python -m picasso ..``, e.g. ``python -m picasso render`` for the render module

Using GPU for Fitting (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To enable GPU fitting, follow instructions on `Gpufit <https://github.com/gpufit/Gpufit>`__ to install the Gpufit python library in your conda environment. In practice, this means downloading the zipfile and installing the Python wheel. Picasso Localize will automatically import the library if present and enables a checkbox for GPU fitting when selecting the LQ-Method.

Example Usage
-------------
Besides using the GUI, you can use picasso like any other Python module. Consider the following example:::

  from picasso import io, postprocess

  path = 'testdata_locs.hdf5'
  locs, info = io.load_locs(path)
  # Link localizations and calcualte dark times
  linked_locs = postprocess.link(picked_locs, info, r_max=0.05, max_dark_time=1)
  linked_locs_dark = postprocess.compute_dark_times(linked_locs)

  print('Average bright time {:.2f} frames'.format(np.mean(linked_locs_dark.n)))
  print('Average dark time {:.2f} frames'.format(np.mean(linked_locs_dark.dark)))

This codeblock loads data from testdata_locs and uses the postprocess functions programmatically.

Jupyter Notebooks
-----------------

Check picasso/samples/ for Jupyter Notebooks that show how to interact with the Picasso codebase.


Contributing
------------

If you have a feature request or a bug report, please post it as an issue on the GitHub issue tracker. If you want to contribute, put a PR for it. You can find more guidelines for contributing `here <https://github.com/jungmannlab/picasso/blob/master/CONTRIBUTING.rst>`__. I will gladly guide you through the codebase and credit you accordingly. Additionally, you can check out the ``Projects``-page on GitHub.  You can also contact me via picasso@jungmannlab.org.




Contributions & Copyright
-------------------------

| Contributors: Joerg Schnitzbauer, Maximilian Strauss, Adrian Przybylski, Andrey Aristov, Hiroshi Sasaki, Alexander Auer, Johanna Rahm
| Copyright (c) 2015-2019 Jungmann Lab, Max Planck Institute of
  Biochemistry
| Copyright (c) 2020-2021 Maximilian Strauss
| Copyright (c) 2022 Rafal Kowalewski

Citing Picasso
--------------

If you use picasso in your research, please cite our Nature Protocols publication describing the software.

| J. Schnitzbauer*, M.T. Strauss*, T. Schlichthaerle, F. Schueder, R. Jungmann
| Super-Resolution Microscopy with DNA-PAINT
| Nature Protocols (2017). 12: 1198-1228 DOI: `https://doi.org/10.1038/nprot.2017.024 <https://doi.org/10.1038/nprot.2017.024>`__

Credits
-------

-  Design icon based on “Hexagon by Creative Stalls from the Noun
   Project”
-  Simulate icon based on “Microchip by Futishia from the Noun Project”
-  Localize icon based on “Mountains by MONTANA RUCOBO from the Noun
   Project”
-  Filter icon based on “Funnel by José Campos from the Noun Project”
-  Render icon based on “Paint Palette by Vectors Market from the Noun
   Project”
-  Average icon based on “Layers by Creative Stall from the Noun
   Project”
-  Server icon based on “Database by Nimal Raj from NounProject.com”
