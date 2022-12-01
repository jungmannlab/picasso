Changelog
=========

Last change: 01-DEC-2022 MTS

0.5.5
-----
- Cluster info is saved in ``_cluster_centers.hdf5`` files which are created when ``Save cluster centers`` box is ticked
- Cluster centers contain info about mean frame (saved as ``frame``), standard deviation frame, area/volume and convex hull
- Bug fixes

0.5.1 - 0.5.4
-------------
- Sensitivity is a float number in Server: Watcher
- One-click-installer available
- Bug fixes

0.5.0
-----
- 3D rendering rotation window
- Multiple .hdf5 files can be loaded when using File->Open
- Localizations can be combined when saving
- Render window restart (Remove all localizations)
- Multiple pyplot colormaps available in Render
- View->Files in Render substantially changed (many new colors, close button works, etc)
- Changing Render's FOV with W, A, S and D
- Render's FOV can be numerically changed, saved and loaded in View->Info
- Pick similar is much faster
- Remove localization in picks
- Fast rendering (display a fraction of localizations)
- .txt file with drift can be applied to localizations in Render
- New clustering algorithm (SMLM clusterer)
- Test clusterer window in Render
- Option to calculate cluster centers
- Nearest neighbor analysis in Render
- Numerical filter in Filter
- New file format in Localize - .nd2 
- Localize can read NDTiffStack.tif files
- Docstrings for Render
- `Plugins <https://picassosr.readthedocs.io/en/latest/plugins.html>`_ can be added to all Picasso modules
- Many other improvements, bug fixes, etc.


0.4.6-10
--------
- Logging for Watcher of Picasso Server
- Mode for multiple parameter groups for Watcher
- Fix for installation on Mac systems
- Various bugfixes


0.4.2-5
-------
- Added more docstrings / documentation for Picasso Server
- Import and export for handling IMS (Imaris) files
- Fixed a bug where GPUFit was greyed out, added better installation instructions for GPUfit
- More documentation
- Added dockerfile


0.4.1
-----
- Fixed a bug in installation


0.4.0
-----
-  Added new module "Picasso Server"