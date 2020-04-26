Picasso
=======
.. image:: https://readthedocs.org/projects/picassosr/badge/?version=latest
   :target: https://picassosr.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
   
.. image:: https://github.com/jungmannlab/picasso/workflows/CI/badge.svg
   :target: https://github.com/jungmannlab/picasso/workflows/CI/badge.svg
   :alt: CI
   
.. image:: main_render.png
   :scale: 100 %
   :alt: UML Render view

A collection of tools for painting super-resolution images. The Picasso software is complemented by our `Nature Protocols publication <https://www.nature.com/nprot/journal/v12/n6/abs/nprot.2017.024.html>`__.
A comprehensive documentation can be found here: `Read the Docs <https://picassosr.readthedocs.io/en/latest/?badge=latest>`__.

Installation
------------

Check out the `Picasso release page <https://github.com/jungmannlab/picasso/releases/>`__ to download and run the latest compiled one-click installer for Windows. Here you will also find the Nature Protocols legacy version. For the platform-independent usage of Picasso (e.g., with Linux and Mac Os X) please follow the advanced installation instructions.

Advanced installation for Python programmers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an alternative to the stand-alone program for end-users, Picasso can be installed as a Python package. This is the preferred option to use Picasso’s internal routines in custom Python programs. For windows, one is still possible to use Picasso as an end-user by creating the respective shortcuts. This allows Picasso to be used on the same system by both programmers and end-users.

Requirements
^^^^^^^^^^^^

Python 3.\* (currently tested with 3.7)
'''''''''''''''''''''''''''''''''''''''

We highly recommend the `Anaconda or Miniconda <https://www.continuum.io/downloads>`__ Python distribution which comes with a powerful package manager.

Setting up environment with conda
'''''''''''''''''''''''''''''''''

Sample instructions to create an environment and installation of packages with conda are as follows:

1. Open the console and create a new conda environment: ``conda create --name picasso python=3.7``
2. Activate the environment: ``source activate picasso`` for Linux / Mac Os X or ``activate picasso`` for Windows.
3. Install the necessary packages with conda: ``conda install h5py matplotlib numba numpy scipy pyqt pyyaml scikit-learn colorama tqdm``
4. Additionally install the lmfit package with pip: ``pip install lmfit``
5. (Optional) If you plan to compile your own installer additionally install Pyinstaller: ``pip install pyinstaller``
6. Continue with the installation of Picasso

Note that sometimes outdated packages can cause problems. As of version 0.3.0, Picasso switched from PyQt4 to PyQt5 so make sure to update PyQt. If you experience errors, please check whether your packages have the right version (e.g. see issue #4). Additionally, make sure that you have the default package channel (e.g., see issue #30). Also, note that there is also a requirements.txt which you can use to install all packages with pip (``pip install -r requirements.txt``). In the past, not relying on conda caused troubles with creating the one-click installer. It is therefore recommended installing all packages with conda.

.. _installation-1:

Installation
^^^^^^^^^^^^

1. Open the console, ``cd`` to the directory where you want to install and run ``git clone https://github.com/jungmannlab/picasso``. Alternatively, `download <https://github.com/jungmannlab/picasso/archive/master.zip>`__ the zip file and unzip it.
2. Change to the downloaded directory ``cd picasso``
3. Run installation ``python setup.py install``.


Optional for windows users
^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the PowerShell script “createShortcuts.ps1” in the gui directory. This should be doable by right-clicking on the script and choosing “Run with PowerShell”. Alternatively, run the command
``powershell ./createShortcuts.ps1`` in the command line. Use the generated shortcuts in the top level directory to start GUI components. Users can drag these shortcuts to their Desktop, Start Menu or Task Bar.

Using Picasso as a module
^^^^^^^^^^^^^^^^^^^^^^^^^

The individual modules of picasso can be started as follows:
1. Open the console, activate the environment: ``source activate picasso`` for Linux / Mac Os X or ``activate picasso`` for Windows.
2. Start the picasso modules via ``python -m picasso ..``, e.g. ``python -m picasso render`` for the render module

Using GPU for Fitting
^^^^^^^^^^^^^^^^^^^^^

To enable GPU fitting, follow instructions on `Gpufit <https://github.com/gpufit/Gpufit>`__ to install the Gpufit python library in your conda environment. Picasso Localize will automatically import the library if present and enables a checkbox for GPU fitting when selecting the LQ-Method.


Contributing
------------

If you have a feature request or a bug report, please post it as an issue on the GitHub issue tracker. If you want to contribute, put a PR for it. You can find more guidelines for contributing `here <https://github.com/jungmannlab/picasso/blob/master/CONTRIBUTING.rst>`__. I will gladly guide you through the codebase and credit you accordingly. Additionally, you can check out the ``Projects``-page on GitHub.  You can also contact me via picasso@jungmannlab.org.

Contributions & Copyright
-------------------------

| Contributors: Joerg Schnitzbauer, Maximilian Strauss, Adrian Przybylski, Andrey Aristov, Hiroshi Sasaki, Alexander Auer, Johanna Rahm
| Copyright (c) 2015-2019 Jungmann Lab, Max Planck Institute of
  Biochemistry

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
