One-click installer for macOS
=============================

This is the one-click installer for Picasso on macOS. Please visit our Github repository (https://github.com/jungmannlab/picasso) for details.

The Picasso software is complemented by our Nature Protocols publication (https://doi.org/10.1038/nprot.2017.024).

A comprehensive documentation can be found here: https://picassosr.readthedocs.io/en/latest/

If something does not work, see the answers to common problems: https://picassosr.readthedocs.io/en/latest/faq.html

How to install
--------------

1. Download the latest release from the release page: https://github.com/jungmannlab/picasso/releases/
2. Open the downloaded dmg file and drag the "picasso" icon to your Applications folder first.
3. Repeat the same for all other icons with separate Picasso modules (Localize, Render, etc) that you wish to use.
4. Note: The main picasso app must remain in the Applications folder, the shortcuts for the other modules can be moved to the desktop or elsewhere if desired.

[!] Picasso is distributed without an Apple Developer ID. macOS may therefore refuse to open the installer, warning that it cannot check it for malicious software. If this happens, click the "?" button in the warning dialog and follow the instructions to open Picasso anyway. This needs to be repeated for both the main "picasso" app and each individual module you use (e.g., Picasso Render). For more details, see: https://support.apple.com/en-gb/guide/mac-help/mh40616/mac

Alternatively, you can create the dmg file yourself by cloning our GitHub repo and running the bash script picasso/release/one_click_macos_gui/create_macos_dmg.sh from the Terminal. Note that you must have conda installed on your computer.

How to update
-------------

When a new version is out, Picasso tells you about it (at most once a day). Click "Install Now" to download it, check it and install it; Picasso closes, replaces Picasso.app and the module apps in the same folder (e.g., Picasso Render) and opens again. Please close other Picasso programs and save your work first. If the apps cannot be replaced, e.g., without the rights to change the Applications folder, the disk image opens so that you can drag the apps to Applications as on the first installation. Settings, camera configuration and plugins in ~/.picasso are kept either way.

Adding camera configuration and plugins
---------------------------------------

Since version 0.11, both the camera configuration and plugins live in your Picasso user folder, ~/.picasso (that is, /Users/<your user name>/.picasso). This is the same folder that already holds settings.yaml and the log file, and it is the same for every installation type (one-click installer, PyPI, conda, source). Because it sits outside the app bundle, it survives updating or removing Picasso, and you no longer need "Show Package Contents" to get to it.

Camera configuration is essential for correct photon conversion and thus correct localization precision calculation. Put your config file at ~/.picasso/config.yaml. The file is never created for you - you have to create it yourself. The quickest way to get there is to open Picasso: Localize and select File > "Open camera config file location", which opens the folder in Finder (creating it if needed) or reveals wherever a config already in use actually lives. To start from a template, copy config_template.yaml (bundled inside the picasso package, next to __init__.py) into ~/.picasso and rename it to config.yaml. For more details, see documentation: https://picassosr.readthedocs.io/en/latest/localize/camera.html#localize-camera-config

Older versions read config.yaml from inside the app bundle (Contents/Frameworks/picasso). *That still works*: if no ~/.picasso/config.yaml exists, Picasso falls back to the in-bundle file and reads it in place, so an existing setup keeps working. When both are present, ~/.picasso/config.yaml wins.

Plugins go in ~/.picasso/plugins, which is created automatically the first time you run any Picasso app. Open it from any Picasso app via Plugins > "Open plugins folder...". A plugin file copied in by hand is found but not enabled: review it and tick its "Enabled" checkbox under Plugins > "Browse online plugins...", which is also where you can install hash-verified plugins from our online registry. The same can be done without a GUI using the "picasso plugins" command. For more details on how to create and manage plugins, see documentation: https://picassosr.readthedocs.io/en/latest/plugins.html

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
- AIM undrifting. Adapted from the authors' code, see the AIM license (https://github.com/jungmannlab/picasso/blob/master/LICENSES/AIM-LICENSE.txt): Ma et al., Sci. Adv., 2024; 10.1126/sciadv.adm7765 (https://www.science.org/doi/10.1126/sciadv.adm7765)
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

-  Design icon based on "Hexagon" by Creative Stall from the Noun Project
-  Simulate icon based on "Microchip" by Futishia from the Noun Project
-  Localize icon based on "Mountains" by MONTANA RUCOBO from the Noun Project
-  Filter icon based on "Funnel" by José Campos from the Noun Project
-  Render icon based on "Paint Palette" by Vectors Market from the Noun Project
-  Average icon based on "Layers" by Creative Stall from the Noun Project
-  Server icon based on "Database" by Nimal Raj from the Noun Project
-  SPINNA icon based on "Spinner" by Viktor Ostrovsky from the Noun Project
-  The Noun Project icons above are used under the CC BY 3.0 (https://creativecommons.org/licenses/by/3.0/) license.
-  Toolbar and menu icons from Lucide (https://lucide.dev) (ISC license). License can be found here (https://github.com/jungmannlab/picasso/blob/master/LICENSES/Lucide-LICENSE.txt).

.. SYNC-END: credits

Third-party licenses
--------------------

The installer bundles Python, Qt (PyQt6) and many other open-source libraries. Their licenses and copyright notices are collected in THIRD-PARTY-LICENSES.txt, on the disk image and inside Picasso.app (Contents/Resources/THIRD-PARTY-LICENSES.txt).
