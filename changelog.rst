Changelog
=========

Last change: 24-FEB-2026 CEST

0.9.8
-----
- Added a function ``picasso.lib.get_save_filename_ext_dialog`` that can also check for the existence of the files with other extenstions (for example, if the user tries to save a .yaml file with the same name as an existing .hdf5 file, it will ask if the user wants to overwrite the .hdf5 file). This is implemented in all GUI modules when saving files.

0.9.7
-----
Important updates:
^^^^^^^^^^^^^^^^^^
- Windows one-click-installer allows for selecting only a subset of Picasso modules to install
- Added ToRaw and Nanotron to one-click-installer
- *Experimental* One-click-installer for macOS (only for Apple Silicon), see `here <https://github.com/jungmannlab/picasso/tree/master/release/one_click_macos_gui>`__

Small improvements:
+++++++++++++++++++
- Adjusted the ``config.yaml`` and plugins instructions for the one-click-installer Picasso release (new Pyinstaller stores everything in the ``_internal`` folder)
- G5M output can save more columns (if present in the input localizations)
- Further enhancement of G5M documentation
- Render GUI: implemented filter by number of localizations for multichannel data
- Render GUI: allow removal of any column from localizations, not only ``group``
- Filter GUI: allow removal of any column from localizations
- ``REQUIRED_COLUMNS`` moved from ``picasso.localize`` to ``picasso.lib``

Bug fixes:
++++++++++
- Fixed basic frame analysis in SMLM clusterer
- Fixed labels of the vertical lines in the subcluster test plot
- Fixed automatic Localize loading/unloading z-calibration paths when changing cameras
- Fixed ``rel_sigma_z`` in G5M (previously incorrectly divided by pixel size)
- Fixed G5M molmap ``lpz`` output
- Fixed loading square picks in Render
- Fixed appearance of the Apply expression dialog in Render for files with many columns
- Fixed initial x, y and N in LQ Gaussian fitting (might results in faster convergence and slightly different (<< NeNA) results) (#616)
- Fixed picking circular regions around left and top edges of the FOV

0.9.6
-----
Important updates:
^^^^^^^^^^^^^^^^^^
- Test subclustering plot (saved after G5M, can be plotted in Filter): fixed the labels of the plots
- Change of API in ``picasso.postprocess.nn_analysis``: new inputs cause backward compatibility issues. The function now returns only the nearest neighbor distances, not the indices of the nearest neighbors.

Small improvements:
+++++++++++++++++++
- Moved from merge sort to quick sort (usually faster due to lower memory usage)
- Render: increase the speed of picking circular locs, picking similar and filter by number of localizations (numba implementation)
- Render property histogram shown before rendering is activated
- Render property - removed legend
- Render Nearest Neighbor Analysis - saves nearest neighbors distances in the localizations .hdf5 file
- Render G5M dialog - adjusted the frame analysis checkbox
- Render G5M: removed the check for min. locs
- Render G5M: moved the check for too large clusters (or if any are present) before applying G5M to all channels (all channels analysis)
- Render masking: mask out saved area uses previously saved area if available in the metadata
- G5M documentation has been updated to include more troubleshooting tips and common issues, see `here <https://picassosr.readthedocs.io/en/latest/render.html#g5m>`__
- Localize zooms in and out centered at the current view
- Config file changes from 0.9.5 were `documented <https://picassosr.readthedocs.io/en/latest/localize.html>`__ and `config template <https://github.com/jungmannlab/picasso/blob/master/picasso/config_template.yaml>`__ was updated
- SPINNA 3D masking: z slicing added for visual inspection
- SPINNA 3D homogeneous simulations automatically adjusts the observed density based on the z range set by the user and the xy area of the pick (if provided)
- SPINNA allows for different mask bin size and blur in lateral and axial dimensions
- SPINNA default mask blur of 500 nm in the API (previously 65 nm)
- Reduced copying and conversion of DataFrames to numpy arrays (less memory usage)
- ``picasso.io.load_locs`` and ``save_locs`` ensure that the saved metadata contains the required keys
- Updated documentation on filetypes and minimum requirements for HDF5 files and accompanying YAML metadata files in Picasso
- Use ``"col" in df.columns`` instead of ``hasattr(df, "col")`` to check for columns in DataFrames (better readability)
- ``picasso.postprocess`` functions ``picked_locs`` and ``pick_similar`` accept precomputed index blocks to speed up the picking of circular regions
- One-click-installer's dependency on ``pkg_resources`` removed (since it has been removed from ``setuptools``)
- Onc-click-installer: PyImarisWriter temporarily removed (caused problems with this release)

Bug fixes:
++++++++++
- SPINNA 3D mask generation fixed (and ``picasso.render.render_hist3d``)
- Test subcluster fix indexing
- Remove backward incompatible camera pixel size reading in SPINNA's mask generation (related to #602)
- Fixed localization masking for non-square mask (``picasso.masking.mask_locs``)
- Correct axial localization precision in Localize (magnification factor)
- Localize does not raise an error if QE is not found in the config file
- Localize does not automatically fit z coordinates if a 3D calibration file is loaded from the config file
- Render Test Clustering: fixed the full FOV button
- Fixed CLI ``picasso join``

0.9.4-5
-------
Important updates:
^^^^^^^^^^^^^^^^^^
- **Algorithm for molecular mapping introduced (G5M)**, see documentation `here <https://picassosr.readthedocs.io/en/latest/render.html#g5m>`__ *DOI will be added once available*.
- **Localize outputs axial localization precision for astigmatic imaging in 3D**,  *DOI will be added once available*. 
- Localize GUI allows the user to select which localization columns to save when saving localizations. See the new dialog in the *File* -> *Select columns to save*
- Localize accepts frame bounds to analyze only a subset of frames
- Config file accepts z calibration .yaml paths so that they can be automatically loaded when changing between cameras
- Render by property (GUI) shows histogram of the selected property
- Filter GUI has a new plot to test for subclustering based on the number of events per molecule (column ``n_events``); see the `Filter documentation <https://picassosr.readthedocs.io/en/latest/filter.html>`__  for details

*Small improvements:*
+++++++++++++++++++++
- Picasso applies constrained layout to all matplotlib figures
- SPINNA uses ``FigureCanvas`` instead of ``QSvgRenderer`` for displaying NND plots and mask legend
- SPINNA default mask blur set to 500 nm (GUI)
- 3D animation saves metadata
- Some improvements in how DataFrames are handled (Filter, change from ``.values`` to ``.to_numpy()``)

*Bug fixes:*
++++++++++++
- Render GUI takes camera pixel size using ``lib.get_from_metadata`` (#602)
- Render by property is switched off if more than one channel is loaded
- Render 3D scale bar manual adjustment fixed
- Render 3D screenshot .yaml fixed
- .tif IO bug fix related to the numpy deprecation of ``arr.newbyteorder`` (#603)
- Clarify GPU fit installation instructions and remove version printing (#604)
- SPINNA fixed loading of the proportion spin boxes after rerunning SPINNA, such that they add up to 100% again

0.9.3
-----
Important updates:
^^^^^^^^^^^^^^^^^^
- All GUI modules show the explanations of parameters when hovering over them with the mouse cursor (tool tips)
- FRC: does not blur rendered localizations, enabled saving rendered images
- Automatic testing at pull requests extended to most Picasso functions

*Small improvements:*
+++++++++++++++++++++
- General improvements in the GUI widget names displayed (for example, change "Scalebar" to "Scale bar")
- Render: many input variables were switched from cam. pixels to nm in the GUI, for example, min. blur in the display settings dialog
- Render: slicer dialog automatically slices/unslices localizations when opening/closing the dialog
- Clustering algorithms copy the input localizations to avoid modifying the input DataFrame (for example, when using Picasso as a package)
- MLE Gauss fitting: default method is now ``sigmaxy``, i.e., sigma can vary between x and y, like in the least-squares fitting
- Upgrade PyPI release action to release/v1 (security reasons)

*Bug fixes:*
++++++++++++
- Average: fix ``pandas`` warnings
- Localize: picasso.localize.identify accepts roi as input argument
- Render: show histogram in mask dialog ignores zero values
- Render: qPAINT histograms in the info dialog fixed and improved
- Fixed ``picasso.postprocess.compute_local_density``

0.9.2
-----
Important updates:
^^^^^^^^^^^^^^^^^^
- Improved and updated `sample notebooks <https://github.com/jungmannlab/picasso/tree/master/samples>`__.
- Render: FRC resolution implementation, see DOI: `10.1038/nmeth.2448 <https://doi.org/10.1038/nmeth.2448>`__. It is calculated for a currently loaded FOV and only one repeat is done. *The exact implementation may change in the future versions.*

*Small improvements:*
+++++++++++++++++++++
- ``picasso.lib.get_from_metadata`` function now has an option to raise a KeyError if the key is not found
- CMD: added undrift by fiducials (``picasso undrift_fiducials``)
- CMD: cleaned up .hdf5 conversion functions (``picasso hdf2csv``, ``picasso csv2hdf`` and `more <https://picassosr.readthedocs.io/en/latest/cmd.html>`__)
- The above functions were moved to ``picasso.io`` module (previously only in ``picasso.gui.render``)
- Picasso: Average CMD was removed since no functionality was implemented

*Bug fixes:*
++++++++++++
- AIM (``picasso.aim.aim``) copies localizations to avoid modifying the input DataFrame.
- AIM: fixed progress bar when no progress object is provided
- Localize: fixed CMD with GPUFit
- Simulate: fixed repetead axes tick labels
- SPINNA: fixed NND plot showing bins/lines outside of xlim
- SPINNA: extract the picked area based on the last .yaml file entry, not the first one (fixes the issue of incorrect densities extracted for localizations that were picked multiple times)
- SPINNA: enforce repeated generation of the search space when exp. data/densities/masks change
- CMD: pair correlation fixed (#588)

0.9.0-1
-------
Important updates:
^^^^^^^^^^^^^^^^^^

- Picasso does not use ``numpy.recarray`` objects anymore. ``pandas.DataFrame`` are used instead. This applies to localizations, drift data, cluster centers, etc. **This change may cause backward compatibility issues when using Picasso as a package (downloaded from PyPI).**
- Updated other dependencies, most importantly, ``numpy`` is now in version 2
- Old setup files were replaced by ``pyproject.toml`` for building and packaging Picasso
- New option to save cluster areas/volumes in DBSCAN, HDBSCAN and SMLM clusterer using Otsu thresholding of rendered images
- Localize: ensure that 3D calibration is centered at z = 0; this guarantees the correct z scaling (magnification factor)
- Render: unfold groups was removed as it is contained within the square grid unfolding
- Render: new pick shape - square
- Render: synchronize groups across channels - removes localizations from groups that are not present in all channels, e.g., after filtering cluster centers by frame analysis, the cluster localizations corresponding to removed cluster centers are also removed
- Render: save pick properties extended to saving group properties, also qpaint index is saved
- SPINNA: improved saved fit results summary (see issue #560)

*Small improvements:*
+++++++++++++++++++++

- Black-based code formatting applied to all scripts
- Cleaned up code for adjusting the size of QWidgets
- Progress dialog shows remaining time estimate more accurately (ignores the offset due to, for example, multiprocessing startup time)
- Render: save pick/group properties saves qpaint index (1 / mean dark time)
- Render: clustering metadata saves fraction of rejected localizations
- Render: screenshot .yaml files can be dragged and dropped to load the display settings
- Render: DBSCAN clustering .yaml file saves min. number of localizations per cluster
- Render 3D: display adjusted after changing blur method
- Localize: localization precision formula for least-squares fitting was corrected to account for a diagonal covariance Gaussian (background term is affected); the function for localization precision was moved from ``picasso.postprocess`` to ``picasso.gausslq``
- SPINNA: GUI single sim does not allow the sum of proportions to exceed 100% (see issue #560)
- SPINNA: save last opened folder added
- SPINNA: smaller font size in NND plot for better readability
- SPINNA: clean up progress dialog
- SPINNA: NN plotting is normalized to 1000 nm
- Simplify the API for picking similar in ``picasso.postprocess``

*Bug fixes:*
++++++++++++
- Render: unfold groups/picks (rectangular grid) fixed for nonconsecutive grouping (the grid might have had missing elements before)
- Render: apply drift from external file fixed
- Render: fix masking (issue #560)
- Render: fix loading camera pixel size from metadata (see issue #560)
- Render: saving picks separately fixed areas in the .yaml files
- Render: loading a new channel with rendering by property fixed
- Render: mouse events are ignored if no localizations are loaded
- Render 3D: remove measurement points fixed
- Render 3D: save rotated localizations fixed
- Render 3D: fixed ind. loc. prec.
- Render 3D: rendering an empty pick fixed
- Localize: user-friendly display of large numbers (for example, 1,052,102 instead of 1052102)
- Localize: fixed acquisition comment extraction from uManager .tif files
- SPINNA: fixed all the bugs related to masking and search space generation (see issue #560)
- SPINNA: save NND plot fixed (when no simulations were run)
- SPINNA: read camera pixel size from metadata fixed (if available)

0.8.8
-----
- Render - masking dialog changed - threshold methods implemented, histogram of values shown, real-time rendering and different dialog layout
- Render - unfolding groups works without the Picasso: Average step beforehand
- Other bug fixes and minor improvements

0.8.5-7
-------
- Sound notifications when long processes finish, see `here <https://picassosr.readthedocs.io/en/latest/others.html>`_
- Several dialogs in Render, Localize and Simulate are now scrollable (*experimental*)
- SPINNA fix automatic area detection from picked localizations
- Render add dependency ``imageio[ffmpeg]`` for building animations
- Render allow for loading pick regions by dropping a .yaml file onto the window
- Render improve zooming with mouse wheel (Ctrl/Cmd + wheel)
- Fast rendering automatically adjusts constrast
- Localize show scale bar function added
- Localize plotted ROI remains the same when zooming in/out and panning
- Localize Gauss MLE saves number of iterations and fit log-likelihood
- DBSCAN accepts min. no. of localizations per cluster
- Cluster center calculations calculate arithmetic mean, not weighted mean
- Other bug fixes and minor improvements

0.8.4
-----
- SPINNA - easy fitting of labeling efficiency
- GUI docstrings added in all scripts; cleaned up docstrings in Picasso modules
- Render: pick size chosen in nm, not camera pixels
- Code clean up (flake8 compliant)
- Other bug fixes

0.8.3
-----
- Design: fix export plates and pipetting schemes
- Design: set default biotin excess to 25 (previously set to 1)
- Render by property allows different colormaps
- Removed ``lmfit`` dependency
- Fix cluster centers bug from v0.8.2

0.8.2
-----
- Added docstrings and data types in all modules (``postprocess``, ``simulate``, ``render``, ``nanotron``, ``localize``, ``lib``, ``io``, ``imageprocess``, ``gaussmle``, ``gausslq``, ``design``, ``clusterer``, ``aim``, ``avgroi`` and ``zfit``)
- Fix one click installer issues for non-administrator users
- Render allows for saving picked localizations in a separate file for each pick
- Remaining time estimate in the progress dialog
- Fix garbage collection when openinging ``.nd2`` files in Localize
- Fix 3D rotation window for a polygon pick
- Render minimap - the zoom-in window is always visible
- Other small fixes and improvements

0.8.1
-----
- Added ``n_events`` to cluster centers, i.e., number of binding events per cluster
- .yaml files contain Picasso version number for easier tracking
- Improved fiducial picking
- Bug fixes and other cosmetic changes

0.8.0
-----
- **New module SPINNA for investigating oligormerization of proteins** , `DOI: 10.1038/s41467-025-59500-z <https://doi.org/10.1038/s41467-025-59500-z>`_
- **NeNA bug fix - old values were (usually) too high by a ~sqrt(2)**
- NeNA bug fix - less prone to fitting to local maximum leading to incorrect values
- NeNA plot - displays distances in nm
- Fiducial picking - filter out picks too few localizations (80% of the total acquisition time)
- ``picasso csv2hdf`` uses pandas to read .csv files
- Bug fixes

0.7.5
-----
- Automatic picking of fiducials added in Render: ``Tools/Pick fiducials``
- Undrifting from picked moved from ``picasso/gui/render`` to ``picasso/postprocess``
- Plugin docs update
- Filter histogram display fixed for datasets with low variance (bug fix)
- AIM undrifting works now if the first frames of localizations are filtered out (bug fix)
- 2D drift plot in Render inverts y axis to match the rendered localizations
- 3D animation fixed
- Other minor bug fixes

0.7.1-4
-------
- SMLM clusterer in picked regions deleted
- Show legend in Render property displayed rounded tick label values
- Pick circular area does not save the area for each pick in localization's metadata 
- Picasso: Render - adjust the scale bar's size automatically based on the current FOV's width
- Picasso: Render - RESI dialog fixed, units in nm
- Picasso: Render - show drift in nm, not camera pixels
- Picasso: Render - masking localizations saves the mask area in its metadata
- Picasso: Render - export current view across channels in grayscale
- Picasso: Render - title bar displays the file only the names of the currently opened files
- CMD implementation of AIM undrifting, see ``picasso aim -h`` in terminal
- CMD localize saves camera information in the metadata file
- Other minor bug fixes

0.7.0
-----
- Adaptive Intersection Maximization (AIM, doi: 10.1038/s41592-022-01307-0) implemented
- Z fitting improved by setting bounds on fitted z values to avoid NaNs
- CMD ``clusterfile`` fixed 
- Picasso: Render 3D, rectangular and polygonal pick fixed
- ``picasso.localize.localize`` fixed
- default MLE fitting uses different sx and sy (CMD only)

0.6.9-11
--------
- Added the option to draw polygon picks in Picasso: Render
- Save pick properties in Picasso: Render saves areas of picked regions in nm^2
- Calibration .yaml file saves number of frames and step size in nm
- ``picasso.lib.merge_locs`` function can merge localizations from multiple files
- Mask dialog in Picasso: Render saves .png mask files
- Mask dialog in Picasso: Render allows to save .png with the blurred image
- Picasso: Localize - added the option to save the current view as a .png file
- Picasso: Render - functions related to picking moved to ``picasso.lib`` and ``picasso.postprocess``
- Picasso: Render - saving picked localizations saves the area(s) of the picked region(s) in the metadata file (.yaml)
- Documentation on readthedocs works again

0.6.6-8
-------
- GUI modules display the Picasso version number in the title bar
- Added readthedocs requirements file (only for developers)
- No blur applied when padding in Picasso: Render (increases speed of rendering)
- Camera settings saved in the .yaml file after localization
- Picasso: Design has the speed optimized extension sequences (Strauss and Jungmann, Nature Methods, 2020)
- Change matplotlib backend for macOS (bug fix with some plots being unavailable)
- .tiff files can be loaded to Localize directly, *although the support may limited!*
- Bug fix: build animation does not trigger antivirus, which could delete Picasso (one click installer only)
- Bug fix: 2D cluster centers area and convex hull are saved correctly
- Bug fix: rectangular picks

0.6.3-5
-------
- Dependencies updated
- Bug fixes due to Python 3.10 and PyQt5 (listed below)
- Fix RCC error for Render GUI (one click installer) (remove tqdm from GUI)
- Fix save pick properties bug in Picasso Render GUI (one click installer)
- Fix render render properties bug in Picasso Render GUI (one click installer)
- Fix animation building in Picasso Render GUI (one click installer)
- Fix test clusterer HDBSCAN bug
- Fix .nd2 localized files info loading (full loader changed to unsafe loader)
- Fix rare bug with pick similar zero division error
- Update installation instructions

0.6.2
-----
- Picasso runs on Python 3.10 (jump from Python 3.7-3.8)
- New installation instructions
- Dependencies updated, meaning that M1 should have no problems with old versions of SciPy, etc.
- Localize: arbitrary number of sensitivity categories
- Picasso Render legend displays larger font
- Picasso Render Test Clusterer displays info when no clusters found instead of throwing an error
- Calling clustering functions from ``picasso.clusterer`` does not require camera pixel size. Same applies for the corresponding functions in CMD. *Only if 3D localizations are used, the pixel size must be provided.*
- HDBSCAN is installed by default since it is distributed within the new version of ``scikit-learn 1.3.0``
- Screenshot ``.yaml`` file contains the list of colors used in the current rendering
- Render scale bar allows only integer values (i.e., no decimals)
- Localize .ims file fitting bug solve

0.6.1
-----
- **Measuring in the 3D window (Measure and scale bar) fixed (previous versions did not convert the value correctly)**
- Localize GUI allows for numerical ROI input in the Parameters Dialog
- Allow loading individual .tif files as in Picasso v0.4.11``
- RESI localizations have the new column ``cluster_id``
- Building animation shows progress (Render 3D)
- Export current view in Render saves metadata; An extra image is saved with a scale bar if the user did not set it
- (**Not applicable in 0.6.2**) Clustering in command window requires camera pixel size to be input (instead of inserting one after calling the function)
- Bug fixes

0.6.0
-----
- New RESI (Resolution Enhancement by Sequential Imaging) dialog in Picasso Render allowing for a substantial resolution boost, (*Reinhardt, et al., Nature, 2023.* DOI: 10.1038/s41586-023-05925-9)
- **Remove quantum efficiency when converting raw data into photons in Picasso Localize**
- Input ROI using command-line ``picasso localize``, see `here <https://picassosr.readthedocs.io/en/latest/cmd.html>`_.

0.5.7
-----
- Updated installation instructions
- (H)DBSCAN available from cmd (bug fix)
- Render group information is faster (e.g., clustered data)
- Test Clusterer window (Render) has multiple updates, e.g., different projections, cluster centers display
- Cluster centers contain info about std in x,y and z
- If localization precision in z-axis is provided, it will be rendered when using ``Individual localization precision`` and ``Individual localization precision (iso)``. **NOTE:** the column must be named ``lpz`` and have the same units as ``lpx`` and ``lpy``.
- Number of CPU cores used in multiprocessing limited at 60
- Updated 3D rendering and clustering documentation
- Bug fixes

0.5.5-6
-------
- Cluster info is saved in ``_cluster_centers.hdf5`` files which are created when ``Save cluster centers`` box is ticked
- Cluster centers contain info about group, mean frame (saved as ``frame``), standard deviation frame, area/volume and convex hull
- ``gist_rainbow`` is used for rendering properties
- NeNA can be calculated many times
- Bug fixes

0.5.0-4
-------
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
- Sensitivity is a float number in Server: Watcher
- `Plugins <https://picassosr.readthedocs.io/en/latest/plugins.html>`_ can be added to all Picasso modules
- Many other improvements, bug fixes, etc.


0.4.6-11
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
- Added new module "Picasso Server"