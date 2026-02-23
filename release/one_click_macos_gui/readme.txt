One-click installer for macOS
=============================

This is the one-click installer for Picasso on macOS. The Picasso software is complemented by our `Nature Protocols publication <https://www.nature.com/nprot/journal/v12/n6/abs/nprot.2017.024.html>`__.

A comprehensive documentation can be found here: `Read the Docs <https://picassosr.readthedocs.io/en/latest/?badge=latest>`__.

How to install
--------------

1. Download the latest release from the `releases page <https://github.com/jungmannlab/picasso/releases/>`__.
2. Open the downloaded .dmg file and drag the "picasso" icon to your Applications folder first.
3. Repeat the same for all other icons with separate Picasso moduels (Localize, Render, etc) that you wish to use.
4. **Note:** The main picasso app must remain in the Applications folder, the shortcuts for the other modules can be moved to the desktop or elsewhere if desired.

Picasso is distributed without an Apple Developer ID, which means that macOS may block the installation of the software. If the steps above do not work, please follow the instructions under the link: https://support.apple.com/en-gb/guide/mac-help/mh40616/mac. Alternatively, you can create the dmg file yourself by cloning our GitHub repo and running the bash script picasso/release/one_click_macos_gui/create_macos_dmg.sh from the Terminal. Note that you must have conda installed on your computer.

Adding camera configuration and plugins
---------------------------------------

Camera configuration is essential for correct photon conversion and thus correct localization precision calculation. For more details, see https://picassosr.readthedocs.io/en/latest/localize.html#camera-config.

To add your config.yaml file, navigate to your Applications folder and right-click on the picasso app, then select "Show Package Contents". Add your config file to Contents/Frameworks/picasso.

Similarly, you can add Picasso plugins under the folder Contents/Frameworks/picasso/gui/plugins. For more details on how to create plugins, see https://picassosr.readthedocs.io/en/latest/plugins.html.

Changelog
---------
To see all changes introduced across releases, see `here <https://github.com/jungmannlab/picasso/blob/master/changelog.rst>`_.

Contributions & Copyright
-------------------------

| Contributors: Joerg Schnitzbauer, Maximilian Strauss, Rafal Kowalewski, Adrian Przybylski, Andrey Aristov, Hiroshi Sasaki, Alexander Auer, Johanna Rahm
| Copyright (c) 2015-2025 Jungmann Lab, Max Planck Institute of Biochemistry
| Copyright (c) 2020-2021 Maximilian Strauss

Citing Picasso
--------------

If you use Picasso in your research, please cite our Nature Protocols publication describing the software.

| J. Schnitzbauer*, M.T. Strauss*, T. Schlichthaerle, F. Schueder, R. Jungmann
| Super-Resolution Microscopy with DNA-PAINT
| Nature Protocols (2017). 12: 1198-1228 DOI: `10.1038/nprot.2017.024 <https://doi.org/10.1038/nprot.2017.024>`__
|
| If you use some of the functionalities provided by Picasso, please also cite the respective publications:

- NeNA. DOI: `10.1007/s00418-014-1192-3 <https://doi.org/10.1007/s00418-014-1192-3>`__
- FRC. DOI: `10.1038/nmeth.2448 <https://doi.org/10.1038/nmeth.2448>`__
- Theoretical lateral localization precision (Gauss LQ and MLE). DOI: `10.1038/nmeth.1447 <https://doi.org/10.1038/nmeth.1447>`__
- Theoretical axial localization precision (Gauss LQ and MLE). DOI: *DOI will be added once available*
- MLE fitting. DOI: `10.1038/nmeth.1449 <https://doi.org/10.1038/nmeth.1449>`__
- RCC undrifting: DOI: `10.1364/OE.22.015982 <https://doi.org/10.1364/OE.22.015982>`__ 
- AIM undrifting. DOI: `10.1126/sciadv.adm776 <https://www.science.org/doi/10.1126/sciadv.adm7765>`__
- SMLM clusterer. DOIs: `10.1038/s41467-021-22606-1 <https://doi.org/10.1038/s41467-021-22606-1>`__ and `10.1038/s41586-023-05925-9 <https://doi.org/10.1038/s41586-023-05925-9>`__
- DBSCAN: Ester, et al. Inkdd, 1996. (Vol. 96, No. 34, pp. 226-231).
- HDBSCAN. DOI: `10.1007/978-3-642-37456-2_14 <https://doi.org/10.1007/978-3-642-37456-2_14>`__
- RESI. DOI: `10.1038/s41586-023-05925-9 <https://doi.org/10.1038/s41586-023-05925-9>`__
- Nanotron. DOI: `10.1093/bioinformatics/btaa154 <https://doi.org/10.1093/bioinformatics/btaa154>`__
- Picasso: Server. DOI: `10.1038/s42003-022-03909-5 <https://doi.org/10.1038/s42003-022-03909-5>`__
- SPINNA. DOI: `10.1038/s41467-025-59500-z <https://doi.org/10.1038/s41467-025-59500-z>`__
- SPINNA for LE fitting. DOI: `10.1038/s41592-024-02242-5 <https://doi.org/10.1038/s41592-024-02242-5>`__
- G5M. DOI: *DOI will be added once available*

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
