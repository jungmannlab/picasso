One-click installer for Windows
===============================

This is the one-click installer for Picasso on Windows. Please visit our Github repository (https://github.com/jungmannlab/picasso) for details.

The Picasso software is complemented by our Nature Protocols publication (https://www.nature.com/nprot/journal/v12/n6/abs/nprot.2017.024.html).

A comprehensive documentation can be found here: https://picassosr.readthedocs.io/en/latest/

How to install
--------------

1. Download the latest release from the release page: https://github.com/jungmannlab/picasso/releases/
2. Open the downloaded exe file and follow the installation instructions.

[!] If installed in "Program Files", Render and Localize may not be available for non-administrator users. Therefore, we recommend installing Picasso outside of "Program Files". The current default location is C:\Picasso.
[!] Camera config and plugins belong in C:\Users\<your user name>\.picasso, not in the installation folder, see "Adding camera configuration and plugins" below.
[!] Windows Safety features and Windows Defender may ask multiple times for permission during the installation and download.

Creating your own installer
---------------------------

You can create the exe file yourself by cloning our GitHub repo and running the script picasso/release/one_click_windows_gui/create_installer_windows.bat from the Command Prompt. Note that you must have conda installed on your computer.

There are two editions of the installer:
- create_installer_windows.bat builds the standard (CPU) installer (Picasso-Windows-64bit-<version>.exe, installs to C:\Picasso).
- create_installer_windows_cuda.bat builds the CUDA edition (Picasso-Windows-64bit-CUDA-<version>.exe, installs to C:\Picasso-CUDA). It additionally bundles the CUDA runtime (numba-cuda[cu12]) so GPU-accelerated (numba.cuda) code can run. It requires an NVIDIA (CUDA-capable) GPU and produces a noticeably larger installer. The two editions install to separate folders and can coexist. GPU rendering in Picasso: Render (via wgpu, any graphics card) is included in both editions; the CUDA edition is only needed for CUDA-accelerated fitting.

Adding camera configuration and plugins
---------------------------------------

Since version 0.11, both the camera configuration and plugins live in your Picasso user folder, C:\Users\<your user name>\.picasso. This is the same folder that already holds settings.yaml and the log file, and it is the same for every installation type (one-click installer, PyPI, conda, source). Because it sits outside the installation folder, it survives updating or uninstalling Picasso, it is writable without administrator rights, and it is shared by the standard and GPU editions.

Camera configuration is essential for correct photon conversion and thus correct localization precision calculation. Put your config file at C:\Users\<your user name>\.picasso\config.yaml. The file is never created for you - you have to create it yourself. The quickest way to get there is to open Picasso: Localize and select File > "Open camera config file location", which opens the folder in Explorer (creating it if needed) or reveals wherever a config already in use actually lives. To start from a template, copy config_template.yaml (bundled inside the picasso package, next to __init__.py) into that folder and rename it to config.yaml. For more details, see documentation: https://picassosr.readthedocs.io/en/latest/localize.html#camera-config

Older versions read config.yaml from the installation folder (C:\Picasso\_internal\picasso). *That still works*: if no config.yaml exists in the user folder, Picasso falls back to the in-package file and reads it in place, so an existing setup keeps working. When both are present, the one in .picasso wins.

Plugins go in C:\Users\<your user name>\.picasso\plugins, which is created automatically the first time you run any Picasso app. Open it from any Picasso app via Plugins > "Open plugins folder...". A plugin file copied in by hand is found but not enabled: review it and tick its "Enabled" checkbox under Plugins > "Browse online plugins...", which is also where you can install hash-verified plugins from our online registry. The same can be done without a GUI using the "picasso plugins" command. For more details on how to create and manage plugins, see documentation: https://picassosr.readthedocs.io/en/latest/plugins.html

Changelog
---------
To see all changes introduced across releases, see: https://github.com/jungmannlab/picasso/blob/master/changelog.md

.. SYNC-START: contributions

Contributions & Copyright
-------------------------

Contributors: Joerg Schnitzbauer, Maximilian Strauss, Rafal Kowalewski, Adrian Przybylski, Andrey Aristov, Hiroshi Sasaki, Alexander Auer, Johanna Rahm
Copyright (c) 2015-2025 Jungmann Lab, Max Planck Institute of Biochemistry

.. SYNC-END: contributions

.. SYNC-START: citing

Citing Picasso
--------------

If you use Picasso in your research, please cite our Nature Protocols publication describing the software.

J. Schnitzbauer*, M.T. Strauss*, T. Schlichthaerle, F. Schueder, R. Jungmann
Super-Resolution Microscopy with DNA-PAINT
Nature Protocols (2017). 12: 1198-1228 DOI: 10.1038/nprot.2017.024 (https://doi.org/10.1038/nprot.2017.024)

If you use some of the functionalities provided by Picasso, please also cite the respective publications:

- All fitting methods are ports of Gpufit. DOI: 10.1038/s41598-017-15313-9 (https://doi.org/10.1038/s41598-017-15313-9). License can be found here (https://github.com/jungmannlab/picasso/blob/master/LICENSES/Gpufit-LICENSE.txt).
- Experimental PSF (cubic-spline) fitting. DOIs: 10.1038/nmeth.4661 (https://doi.org/10.1038/nmeth.4661) (Li et al., experimental-PSF localization and bead alignment) and 10.1038/s41598-017-00622-w (https://doi.org/10.1038/s41598-017-00622-w) (Babcock & Zhuang, cubic-spline PSF model). The spline calibration follows the coefficient scheme of Gpuspline; license can be found here (https://github.com/jungmannlab/picasso/blob/master/LICENSES/Gpuspline-LICENSE.txt).
- Multichannel (global) experimental-PSF fitting. DOI: 10.1038/s41467-022-30719-4 (https://doi.org/10.1038/s41467-022-30719-4) (Li et al., globLoc).
- 3D fitting via astigmatism. DOI: 10.1126/science.1153529 (https://www.science.org/doi/10.1126/science.1153529).
- B-spline wavelet spot identification. DOI: 10.1364/OE.20.002081 (https://doi.org/10.1364/OE.20.002081) (Izeddin et al., Opt. Express 2012)
- sCMOS pixel-dependent noise modeling. DOI: 10.1038/nmeth.2488 (https://doi.org/10.1038/nmeth.2488).
- NeNA. DOI: 10.1007/s00418-014-1192-3 (https://doi.org/10.1007/s00418-014-1192-3)
- FRC. DOI: 10.1038/nmeth.2448 (https://doi.org/10.1038/nmeth.2448)
- Theoretical lateral localization precision (lpx / lpy, Gaussian least-squares). DOI: 10.1038/nmeth.1447 (https://doi.org/10.1038/nmeth.1447)
- Theoretical axial localization precision (lpz values, Gaussian). DOI: 10.1038/s41467-026-70198-5 (https://doi.org/10.1038/s41467-026-70198-5)
- Quad-tree adaptive histogram rendering. DOI: 10.1017/S143192760999122X (https://doi.org/10.1017/S143192760999122X) (Baddeley, Cannell & Soeller, Microsc. Microanal. 2010)
- RCC undrifting: DOI: 10.1364/OE.22.015982 (https://doi.org/10.1364/OE.22.015982)
- AIM undrifting. DOI: 10.1126/sciadv.adm776 (https://www.science.org/doi/10.1126/sciadv.adm7765)
- SMLM clusterer. DOIs: 10.1038/s41467-021-22606-1 (https://doi.org/10.1038/s41467-021-22606-1) and 10.1038/s41586-023-05925-9 (https://doi.org/10.1038/s41586-023-05925-9)
- DBSCAN: Ester, et al. Inkdd, 1996. (Vol. 96, No. 34, pp. 226-231).
- Anisotropic DBSCAN inspired by: 10.1021/acs.jpcb.4c02030 (https://doi.org/10.1021/acs.jpcb.4c02030)
- HDBSCAN. DOI: 10.1007/978-3-642-37456-2_14 (https://doi.org/10.1007/978-3-642-37456-2_14)
- RESI. DOI: 10.1038/s41586-023-05925-9 (https://doi.org/10.1038/s41586-023-05925-9)
- Nanotron. DOI: 10.1093/bioinformatics/btaa154 (https://doi.org/10.1093/bioinformatics/btaa154)
- Picasso: Server. DOI: 10.1038/s42003-022-03909-5 (https://doi.org/10.1038/s42003-022-03909-5)
- SPINNA. DOI: 10.1038/s41467-025-59500-z (https://doi.org/10.1038/s41467-025-59500-z)
- SPINNA for LE fitting. DOI: 10.1038/s41592-024-02242-5 (https://doi.org/10.1038/s41592-024-02242-5)
- G5M. DOI: 10.1038/s41467-026-70198-5 (https://doi.org/10.1038/s41467-026-70198-5)

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

Creating your own installer
---------------------------

You can create the exe file yourself by cloning our GitHub repo and running the script picasso/release/one_click_windows_gui/create_installer_windows.bat from the Command Prompt. Note that you must have conda installed on your computer.

There are two editions of the installer:
- create_installer_windows.bat builds the standard (CPU) installer (Picasso-Windows-64bit-<version>.exe, installs to C:\Picasso).
- create_installer_windows_gpu.bat builds the GPU edition (Picasso-Windows-64bit-GPU-<version>.exe, installs to C:\Picasso-GPU). It additionally bundles the CUDA runtime (numba-cuda[cu12]) so GPU-accelerated (numba.cuda) code can run. It requires an NVIDIA (CUDA-capable) GPU and produces a noticeably larger installer. The two editions install to separate folders and can coexist.
