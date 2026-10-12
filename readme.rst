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

|logo|

.. |logo| image:: https://raw.githubusercontent.com/jungmannlab/picasso/master/docs/_static/picasso-logo-card.png
   :alt: Picasso logo
   :width: 400

A collection of tools for painting super-resolution images, covering single-molecule localization microscopy (SMLM) analysis from raw movies to localization, rendering and quantification. Picasso is complemented by our `Nature Protocols publication <https://doi.org/10.1038/nprot.2017.024>`__.

**Documentation:** `picassosr.readthedocs.io <https://picassosr.readthedocs.io/en/latest/>`__ - `Installation <https://picassosr.readthedocs.io/en/latest/getting-started/installation.html>`__, `First steps <https://picassosr.readthedocs.io/en/latest/getting-started/workflow.html>`__, `Python API <https://picassosr.readthedocs.io/en/latest/api/index.html>`__, `Changelog <https://github.com/jungmannlab/picasso/blob/master/changelog.md>`__.

Installation
------------

- **One-click installer** (Windows, macOS): download it from the `release page <https://github.com/jungmannlab/picasso/releases/>`__.
- **PyPI**: ``pip install picassosr``, then start a module with ``picasso render``, ``picasso localize``, etc.

Optional extras (file formats, GPU support) and the developer installation are described in the `installation guide <https://picassosr.readthedocs.io/en/latest/getting-started/installation.html>`__.

Example Usage
-------------

Besides the GUI, Picasso can be used like any other Python package::

  from picasso import io, postprocess

  locs, info = io.load_locs("testdata_locs.hdf5")

  # Link localizations and calculate dark times
  linked_locs = postprocess.link(locs, info, r_max=0.05, max_dark_time=1)
  linked_locs_dark = postprocess.compute_dark_times(linked_locs)

  print(f"Average bright time {linked_locs_dark['n'].mean():.2f} frames")
  print(f"Average dark time {linked_locs_dark['dark'].mean():.2f} frames")

For more examples, see the `sample notebooks <https://github.com/jungmannlab/picasso/tree/master/samples>`__ and the `Python API documentation <https://picassosr.readthedocs.io/en/latest/api/index.html>`__.

Contributing
------------

Please post feature requests and bug reports on the `issue tracker <https://github.com/jungmannlab/picasso/issues>`__; pull requests are welcome (see the `contributing guidelines <https://github.com/jungmannlab/picasso/blob/master/CONTRIBUTING.rst>`__). You can also contact us via picasso@jungmannlab.org.

.. SYNC-START: contributions

Contributions & Copyright
-------------------------

| Contributors: Rafal Kowalewski, Maximilian Strauss, Joerg Schnitzbauer, Heinrich Grabmayr, Adrian Przybylski, Alexander Auer and `others <https://github.com/jungmannlab/picasso/graphs/contributors>`__ 
| Copyright (c) 2015-2026 Jungmann Lab, Max Planck Institute of Biochemistry

.. SYNC-END: contributions

.. SYNC-START: citing

Citing Picasso
--------------

If you use Picasso in your research, please cite our Nature Protocols publication describing the software.

| J. Schnitzbauer*, M.T. Strauss*, T. Schlichthaerle, F. Schueder, R. Jungmann
| Super-Resolution Microscopy with DNA-PAINT
| Nature Protocols (2017). 12: 1198-1228 DOI: `10.1038/nprot.2017.024 <https://doi.org/10.1038/nprot.2017.024>`__

Many of the functionalities provided by Picasso were published independently, please also cite the respective publications:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Feature
     - Reference
   * - Fitting (all methods are ports of Gpufit)
     - `10.1038/s41598-017-15313-9 <https://doi.org/10.1038/s41598-017-15313-9>`__; `license <https://github.com/jungmannlab/picasso/blob/master/LICENSES/Gpufit-LICENSE.txt>`__
   * - Experimental PSF (cubic-spline) fitting
     - `10.1038/nmeth.4661 <https://doi.org/10.1038/nmeth.4661>`__ (Li et al., experimental-PSF localization and bead alignment) and `10.1038/s41598-017-00622-w <https://doi.org/10.1038/s41598-017-00622-w>`__ (Babcock & Zhuang, cubic-spline PSF model). The spline calibration follows the coefficient scheme of Gpuspline; `license <https://github.com/jungmannlab/picasso/blob/master/LICENSES/Gpuspline-LICENSE.txt>`__
   * - Multichannel (global) experimental-PSF fitting
     - `10.1038/s41467-022-30719-4 <https://doi.org/10.1038/s41467-022-30719-4>`__ (Li et al., globLoc)
   * - 3D fitting via astigmatism
     - `10.1126/science.1153529 <https://www.science.org/doi/10.1126/science.1153529>`__
   * - B-spline wavelet spot identification
     - `10.1364/OE.20.002081 <https://doi.org/10.1364/OE.20.002081>`__ (Izeddin et al., Opt. Express 2012)
   * - sCMOS pixel-dependent noise modeling
     - `10.1038/nmeth.2488 <https://doi.org/10.1038/nmeth.2488>`__
   * - NeNA
     - `10.1007/s00418-014-1192-3 <https://doi.org/10.1007/s00418-014-1192-3>`__
   * - FRC
     - `10.1038/nmeth.2448 <https://doi.org/10.1038/nmeth.2448>`__
   * - Theoretical lateral localization precision (``lpx`` / ``lpy``, Gaussian least-squares)
     - `10.1038/nmeth.1447 <https://doi.org/10.1038/nmeth.1447>`__
   * - Theoretical axial localization precision (``lpz``, Gaussian)
     - `10.1038/s41467-026-70198-5 <https://doi.org/10.1038/s41467-026-70198-5>`__
   * - Quad-tree adaptive histogram rendering
     - `10.1017/S143192760999122X <https://doi.org/10.1017/S143192760999122X>`__ (Baddeley, Cannell & Soeller, Microsc. Microanal. 2010)
   * - RCC undrifting
     - `10.1364/OE.22.015982 <https://doi.org/10.1364/OE.22.015982>`__
   * - AIM undrifting
     - `10.1126/sciadv.adm7765 <https://www.science.org/doi/10.1126/sciadv.adm7765>`__
   * - SMLM clusterer
     - `10.1038/s41467-021-22606-1 <https://doi.org/10.1038/s41467-021-22606-1>`__ and `10.1038/s41586-023-05925-9 <https://doi.org/10.1038/s41586-023-05925-9>`__
   * - DBSCAN
     - Ester et al., Proc. KDD 1996, Vol. 96, No. 34, pp. 226-231
   * - Anisotropic DBSCAN
     - Inspired by `10.1021/acs.jpcb.4c02030 <https://doi.org/10.1021/acs.jpcb.4c02030>`__
   * - HDBSCAN
     - `10.1007/978-3-642-37456-2_14 <https://doi.org/10.1007/978-3-642-37456-2_14>`__
   * - RESI
     - `10.1038/s41586-023-05925-9 <https://doi.org/10.1038/s41586-023-05925-9>`__
   * - Nanotron
     - `10.1093/bioinformatics/btaa154 <https://doi.org/10.1093/bioinformatics/btaa154>`__
   * - Picasso: Server
     - `10.1038/s42003-022-03909-5 <https://doi.org/10.1038/s42003-022-03909-5>`__
   * - SPINNA
     - `10.1038/s41467-025-59500-z <https://doi.org/10.1038/s41467-025-59500-z>`__
   * - SPINNA for LE fitting
     - `10.1038/s41592-024-02242-5 <https://doi.org/10.1038/s41592-024-02242-5>`__
   * - G5M
     - `10.1038/s41467-026-70198-5 <https://doi.org/10.1038/s41467-026-70198-5>`__
   * - Coordinate-based colocalization (CBC)
     - `10.1007/s00418-011-0880-5 <https://doi.org/10.1007/s00418-011-0880-5>`__

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
-  Toolbar and menu icons from `Lucide <https://lucide.dev>`__ (ISC license). License can be found `here <https://github.com/jungmannlab/picasso/blob/master/LICENSES/Lucide-LICENSE.txt>`__.

.. SYNC-END: credits
