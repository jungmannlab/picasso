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

The two scripts can run at the same time, e.g., in two Command Prompts.

Adding camera configuration and plugins
---------------------------------------

Since version 0.11, both the camera configuration and plugins live in your Picasso user folder, C:\Users\<your user name>\.picasso. This is the same folder that already holds settings.yaml and the log file, and it is the same for every installation type (one-click installer, PyPI, conda, source). Because it sits outside the installation folder, it survives updating or uninstalling Picasso, it is writable without administrator rights, and it is shared by the standard and CUDA editions.

Camera configuration is essential for correct photon conversion and thus correct localization precision calculation. Put your config file at C:\Users\<your user name>\.picasso\config.yaml. The file is never created for you - you have to create it yourself. The quickest way to get there is to open Picasso: Localize and select File > "Open camera config file location", which opens the folder in Explorer (creating it if needed) or reveals wherever a config already in use actually lives. To start from a template, copy config_template.yaml (bundled inside the picasso package, next to __init__.py) into that folder and rename it to config.yaml. For more details, see documentation: https://picassosr.readthedocs.io/en/latest/localize.html#camera-config

Older versions read config.yaml from the installation folder (C:\Picasso\_internal\picasso). *That still works*: if no config.yaml exists in the user folder, Picasso falls back to the in-package file and reads it in place, so an existing setup keeps working. When both are present, the one in .picasso wins.

Plugins go in C:\Users\<your user name>\.picasso\plugins, which is created automatically the first time you run any Picasso app. Open it from any Picasso app via Plugins > "Open plugins folder...". A plugin file copied in by hand is found but not enabled: review it and tick its "Enabled" checkbox under Plugins > "Browse online plugins...", which is also where you can install hash-verified plugins from our online registry. The same can be done without a GUI using the "picasso plugins" command. For more details on how to create and manage plugins, see documentation: https://picassosr.readthedocs.io/en/latest/plugins.html

Changelog
---------
To see all changes introduced across releases, see: https://github.com/jungmannlab/picasso/blob/master/changelog.md

.. SYNC-START: contributions

Contributions & Copyright
-------------------------

Contributors: Rafal Kowalewski, Maximilian Strauss, Joerg Schnitzbauer, Heinrich Grabmayr, Adrian Przybylski, Alexander Auer and others (https://github.com/jungmannlab/picasso/graphs/contributors) 
Copyright (c) 2015-2026 Jungmann Lab, Max Planck Institute of Biochemistry

.. SYNC-END: contributions

.. SYNC-START: citing

Citing Picasso
--------------

If you use Picasso in your research, please cite our Nature Protocols publication describing the software.

J. Schnitzbauer*, M.T. Strauss*, T. Schlichthaerle, F. Schueder, R. Jungmann
Super-Resolution Microscopy with DNA-PAINT
Nature Protocols (2017). 12: 1198-1228 DOI: 10.1038/nprot.2017.024 (https://doi.org/10.1038/nprot.2017.024)

Many of the functionalities provided by Picasso were published independently, please also cite the respective publications:

- Fitting. All methods are ports of Gpufit, see the Gpufit license (https://github.com/jungmannlab/picasso/blob/master/LICENSES/Gpufit-LICENSE.txt): Przybylski et al., Sci. Rep., 2017; 10.1038/s41598-017-15313-9 (https://doi.org/10.1038/s41598-017-15313-9)
- Experimental PSF (cubic-spline) fitting. The calibration follows the coefficient scheme of Gpuspline, see the Gpuspline license (https://github.com/jungmannlab/picasso/blob/master/LICENSES/Gpuspline-LICENSE.txt): Li et al., Nat. Methods, 2018; Babcock & Zhuang, Sci. Rep., 2017; 10.1038/nmeth.4661 (https://doi.org/10.1038/nmeth.4661), 10.1038/s41598-017-00622-w (https://doi.org/10.1038/s41598-017-00622-w)
- Multichannel (global) experimental-PSF fitting (globLoc): Li et al., Nat. Commun., 2022; 10.1038/s41467-022-30719-4 (https://doi.org/10.1038/s41467-022-30719-4)
- 3D fitting via astigmatism: Huang et al., Science, 2008; 10.1126/science.1153529 (https://www.science.org/doi/10.1126/science.1153529)
- B-spline wavelet spot identification: Izeddin et al., Opt. Express, 2012; 10.1364/OE.20.002081 (https://doi.org/10.1364/OE.20.002081)
- sCMOS pixel-dependent noise modeling: Huang et al., Nat. Methods, 2013; 10.1038/nmeth.2488 (https://doi.org/10.1038/nmeth.2488)
- NeNA: Endesfelder et al., Histochem. Cell Biol., 2014; 10.1007/s00418-014-1192-3 (https://doi.org/10.1007/s00418-014-1192-3)
- FRC: Nieuwenhuizen et al., Nat. Methods, 2013; 10.1038/nmeth.2448 (https://doi.org/10.1038/nmeth.2448)
- Theoretical lateral localization precision (lpx / lpy, Gaussian least-squares): Mortensen et al., Nat. Methods, 2010; 10.1038/nmeth.1447 (https://doi.org/10.1038/nmeth.1447)
- Theoretical axial localization precision (lpz, Gaussian): Kowalewski et al., Nat. Commun., 2026; 10.1038/s41467-026-70198-5 (https://doi.org/10.1038/s41467-026-70198-5)
- Quad-tree adaptive histogram rendering: Baddeley et al., Microsc. Microanal., 2010; 10.1017/S143192760999122X (https://doi.org/10.1017/S143192760999122X)
- RCC undrifting: Wang et al., Opt. Express, 2014; 10.1364/OE.22.015982 (https://doi.org/10.1364/OE.22.015982)
- AIM undrifting: Ma et al., Sci. Adv., 2024; 10.1126/sciadv.adm7765 (https://www.science.org/doi/10.1126/sciadv.adm7765)
- SMLM clusterer: Schlichthaerle et al., Nat. Commun., 2021; Reinhardt et al., Nature, 2023; 10.1038/s41467-021-22606-1 (https://doi.org/10.1038/s41467-021-22606-1), 10.1038/s41586-023-05925-9 (https://doi.org/10.1038/s41586-023-05925-9)
- DBSCAN: Ester et al., KDD, 1996 (Vol. 96, No. 34, pp. 226-231)
- Anisotropic DBSCAN (inspired by): Lörzing et al., J. Phys. Chem. B, 2024; 10.1021/acs.jpcb.4c02030 (https://doi.org/10.1021/acs.jpcb.4c02030)
- HDBSCAN: Campello et al., PAKDD, 2013; 10.1007/978-3-642-37456-2_14 (https://doi.org/10.1007/978-3-642-37456-2_14)
- RESI: Reinhardt et al., Nature, 2023; 10.1038/s41586-023-05925-9 (https://doi.org/10.1038/s41586-023-05925-9)
- Nanotron: Auer et al., Bioinformatics, 2020; 10.1093/bioinformatics/btaa154 (https://doi.org/10.1093/bioinformatics/btaa154)
- Picasso Server: Strauss, Commun. Biol., 2022; 10.1038/s42003-022-03909-5 (https://doi.org/10.1038/s42003-022-03909-5)
- SPINNA: Masullo et al., Nat. Commun., 2025; 10.1038/s41467-025-59500-z (https://doi.org/10.1038/s41467-025-59500-z)
- SPINNA for labeling efficiency (LE) fitting: Hellmeier et al., Nat. Methods, 2024; 10.1038/s41592-024-02242-5 (https://doi.org/10.1038/s41592-024-02242-5)
- G5M: Kowalewski et al., Nat. Commun., 2026; 10.1038/s41467-026-70198-5 (https://doi.org/10.1038/s41467-026-70198-5)
- Coordinate-based colocalization (CBC): Malkusch et al., Histochem. Cell Biol., 2012; 10.1007/s00418-011-0880-5 (https://doi.org/10.1007/s00418-011-0880-5)
- Ripley's K and H functions: Ripley, J. R. Stat. Soc. B, 1977; 10.1111/j.2517-6161.1977.tb01615.x (https://doi.org/10.1111/j.2517-6161.1977.tb01615.x)
- Two-channel (cross) Ripley's H and radial distribution functions: Lotwick and Silverman, J. R. Stat. Soc. B, 1982; 10.1111/j.2517-6161.1982.tb01221.x (https://doi.org/10.1111/j.2517-6161.1982.tb01221.x)
- Stitching tiles (global tile positions, inspired by): Preibisch et al., Bioinformatics, 2009; 10.1093/bioinformatics/btp184 (https://doi.org/10.1093/bioinformatics/btp184)
- Single-particle tracking (nearest-neighbor tracking, jump distance and MSD analysis): Martens et al., Front. Bioinform., 2021; 10.3389/fbinf.2021.817254 (https://doi.org/10.3389/fbinf.2021.817254)

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
-  Toolbar and menu icons from Lucide (https://lucide.dev) (ISC license). License can be found here (https://github.com/jungmannlab/picasso/blob/master/LICENSES/Lucide-LICENSE.txt).

.. SYNC-END: credits

Third-party licenses
--------------------

The installer bundles Python, Qt (PyQt6) and many other open-source libraries. Their licenses and copyright notices are collected in THIRD-PARTY-LICENSES.txt, in the installation folder (C:\Picasso, or C:\Picasso-CUDA for the CUDA edition).
