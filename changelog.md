# Changelog

Last change: 19-MAY-2026 CEST

## 0.10.1

- SPINNA: comparing models uses fitting modes and has cleaner progress dialog
- SPINNA: convenient fitting of LE

## 0.10.0

### **General updates:**

- Numerous new functions added in the API to simplify the more complicated analyses, for example, `picasso.localize.fit2D`
- Installing Picasso as a package has less stringent dependencies and Python version requirements, the exact versions are specified for one-click-installers only
- Almost all the functions in the GUI scripts (for example, `picasso.gui.render.py`) not related to GUI were moved to corresponding API scripts such that using Picasso as a Python package allows for easy analysis analogous to what GUI provides. For example, ``picasso.render.py`` does not only provide the function to generate a grayscale image of localizations only (like before) but can also be used to paint the same images with a color map as they are rendered (for example, with picks and scale bar)
- One-click installer uses Python 3.14 (previously 3.10) and updated dependencies, which should improve the performance of some functions
- Easy access to user settings via any Picasso module
- Expanded test suite (CI)
- Picasso automatically checks for updates when launched and notifies the user if a new version is available
- Render, Localize, Average and Filter allow the user to inspect metadata in the app

#### Localize
- [GPUfit](https://github.com/gpufit/Gpufit) incorporated into Picasso (`picasso.ext.pygpufit`)
- Localize supports .stk file format from MetaMorph (*experimental*)
- Abort button to stop asynchronous multiprocessing (for example, during identification)
- Error box compatible with multiprocessed tasks (clear error message)
- Save and load identifications
- Save spots as .tif, .npy, not .hdf5
- Documentation updated relating to the file menu features, such as loading picks as identifications
- Fixed reading .ims movies
- Fixed spot saving
- Export current view is less pixelated
- Localization markers (green crosses) in the GUI are not affected by drift correction (only visual improvement)
- CLI `picasso localize <files>` allows for MLE fitting in 3D (z-fitting still as per Huang et al, 2008.)

#### Render
- Smarter fast rendering, lowering RAM usage almost two-fold
- Faster ind. loc. precision rendering in 3D
- Test clustering supports G5M
- Test clustering saves the channel to which the algorithms are applied
- Test clustering allows for applying the current parameters to the whole dataset
- Test clustering tool tips
- G5M calculates more accurate sigma constraints in 3D
- Plot localizations profile for rectangular pick
- Reading .csv files from ThunderSTORM
- Mask settings dialog allows for zooming and panning
- More accessible saving/loading of FOVs as .txt files
- Show NeNA/FRC plot buttons automatically calculate them if not done already
- Keyboard shortcut for closing all localizations (Ctrl+Shift+Backspace or Ctrl+Shift+Delete)
- Legend is displayed on black background for better visibility
- Log-scaling of contrast
- New image exporting with manually selected rendering options + support for .pdf and .svg formats
- Optimal scale bar is only set upon user's request
- Changed the name "Nearest Neighbor Analysis" to "Calculate nearest neighbor distances" for better clarity
- Faster non-circle picking by smarter indexing
- Trace shows number of photons in addition to x, y and frame; exports .csv files with three columns (frame, ON/OFF and photons)
- Manual setting of scale bar switches off automatic scale bar length
- Apply drift from external file supports dropping the .txt file onto the window
- Show drift keeps x and y coords to scale (on the second plot)
- 3D rotation window supports rendering by property
- More intuitive rotation in 3D instead of simple rotations around xyz axes
- Animation dialog allows unlimited positions
- Fixed 3D animation for non-square FOV
- Fixed distances in NeNA plot (previously plotting multiple times kept increasing the values)
- Fixed panning in 3D
- Fixed 3D screenshot metadata
- Fixed pre-G5M group/max locs checks when applying to all channels
- Fixed zero-value in rendered images (previously RGB channels were capped between 1 and 255 instead of 0 and 255)
- Fixed default directory for applying drift from external file
- Added attribute `pixelsize` in View for cleaner code

#### SPINNA
- Two new fitting methods for fast fitting instead of the brute force search, see [documentation](https://picassosr.readthedocs.io/en/latest/spinna.html#fitting)
- User-defined threshold for the binary mask
- Loading new structures in the Simulate tab without changing targets does not reset the window
- Fixed .svg saving in the one-click-installer app
- Fixed issues caused by removing structures in the Structures Tab (Windows)

#### Filter
- Support for .csv export (not only hdf5)
- Apply filtering steps from metadata
- Filtering range for numerical filtering is inclusive

#### Average
- Abort button
- Improved saved metadata
- Adjusted default parameters

#### *Other improvements:*
- Only `picasso.version.py` determines software version globally, thus `bumpversion` is not needed anymore
- `picasso.lib.merge_locs` allows for flexible `frame` and `group` incrementing when merging localizations lists
- New functions in the API `picasso.postprocess.undrift_from_fiducials` and `picasso.postprocess.apply_drift` that can be used to undrift localizations based on picked fiducials with or without user-specified picks and to apply the calculated drift to the localizations, respectively
- New API for alignement of locs, see ``picasso.postprocess``: ``align_rcc`` and ``align_from_picked``
- New function ``picasso.io.load_picks``
- Adjusted installation instructions in README
- Badges added to the GitHub repository (PyPI and Python versions, changelog)
- Dialogs with scroll areas show no margins (e.g., Display settings dialog in Render)
- Added help buttons to some dialogs/menu bars across the modules that open the corresponding readthedocs pages (the documentation will be further improved in the future)
- "What's this?" help button removed from all dialogs (Windows) as it previously crashed Picasso
- Changelog changed from .rst to markdown for GitHub display
- Removed focus on push buttons in dialogs
- Improved data typing of numpy arrays
- Fixed flake8 warnings (code style only)
- `picasso.postprocess.groupprops` shows no progress by default
- `picasso.io.TiffMultiMap` docstrings corrected
- CLI function `nneighbor` uses KDTree for higher speed
- Picasso: Simulate (multilabel) saves label names as in "Exchange rounds to be simulated" rather than 0, 1, 2, ...
- Fixed Picasso: ToRaw
- `path.replace()` is no longer used to change the extension of the path (safer approach)

### **Backward incompatible changes:**

- Several new depedencies have been added. If Picasso is installed via PyPI (`pip install picassosr`) or one-click-installer, no action needs to be taken. **Otherwise please install them when updating Picasso to v0.10.0**. The dependencies are: `tifffile`, `hdf5plugin` (only for Windows to read .ims files). Additionally `PyQt5` was updated to `PyQt6`.
- `picasso.spinna.SPINNA.fit` accepts all inputs as keyword arguments (except for `N_structures`).
- Names of nearly all functions in `picasso.g5m` and some in `picasso.zfit` have been changed (underscore added to prefix as private functions). The main functions in these scripts were left unchanged: `g5m.g5m`, `zfit.zfit`. Functions `zfit.fit_z` and `zfit.fit_z_parallel` are deprecated, see below.
- Cluster centers (DBSCAN, HDBSCAN, SMLM clusterer) save number of localizations per cluster as `n_locs`, not `n`.

#### *Deprecation warnings:*

- `picasso.lib.unpack_calibration` and the `spot_size`, `z_range` parameters in the G5M functions. `picasso.g5m.g5m` now uses calibration coefficients only for setting sigma constraints in 3D for more accurate results.
- `picasso.clusterer.cluster_center` (will be renamed to `_cluster_center` and become a private function in v0.11.0)
- `picasso.aim`: `intersect1d`, `count_intersections`, `run_intersections`, `run_intersections_multithread`, `get_fft_peak`, `get_fft_peak_z`, `point_intersect_2d` and `point_intersect_3d` (will become private functions in v0.11.0)
- `picasso.masking.mask_locs` uses metadata rather than now deprecated `width` and `height` parameters
- `picasso.spinna.MaskGenerator`: `run_checks` parameter (will be removed in v0.11.0)
- `picasso.localize.identify` and `picasso.localize.localize` will return metadata by default in v0.11.0
- `fit_z` and `fit_z_parallel` in `picasso.zfit` will be deprecated in v0.11.0. `zfit.zfit` takes over as the main function in the script
- `picasso.render` takes in `disp_px_size` rather than `oversampling`, see the function; `oversampling` will be removed in v0.11.0
- `picasso.render` functions: `render_hist`, `render_gaussian`, `render_gaussian_iso`, `render_smooth` and `render_convolve` will become private in v0.11.0
- `picasso.gausslq.initial_parameters_gpufit` and `picasso.gaussmle.mean_filter` will become private in v0.11.0
- `picasso.localize` functions: `local_maxima`, `gradient_at`, `net_gradient` will become private in v0.11.0. Functions `fit` and `fit_async` will be removed entirely
- `picasso.postprocess` functions: `index_blocks_shape`, `n_block_locs_at`, `next_frame_neighbor_distance_histogram`, `get_link_groups` and `link_loc_groups` will become private in v0.11.0
- `picasso.spinna` functions: `find_target_counts`, `get_structures_permutation`, `targets_from_structures`

## 0.9.10

### Important updates:

- Added support for loading BigTIFF in Picasso Localize (#631), big thanks to @boydcpeters

### Small improvements:

- `picasso.aim.aim` accepts progress as a `lib.ProgressDialog`, `"console"` or `None`
- SPINNA GUI: Small adjustment to GUI when loading search space
- Adjusted label in subcluster check plot
- Subcluster check plot outputs p value and test statistic

### Bug fixes:

- Fixed AIM in Localize GUI
- Fixed saving search space in SPINNA for multiple-target structures

## 0.9.8-9

### Small improvements:

- Added a function `picasso.lib.get_save_filename_ext_dialog` that can also check for the existence of the files with other extenstions (for example, if the user tries to save a .yaml file with the same name as an existing .hdf5 file, it will ask if the user wants to overwrite the .hdf5 file). This is implemented in all GUI modules when saving files.
- `PyImarisWriter` is included in the one-click-installer again (Windows only)
- Localize GUI allows the user to automatically undrift localizations
- Localize Parameters dialog displays a message if the z calibration path in the config file could not be found
- MLE fitting saves CRLB uncertainties of fitted parameters: photons, background, sx and sy
- `picasso.localize.fit` default method changed to `sigmaxy` (anisotropic sigma fitting)
- Render export localizations supports exporting all channels sequentially
- Changed default max. frames in linking (dark times calculation) to 3 (previously 1) (both GUI and `picasso.postprocess.link`)
- Added number of binding events to Render's "Show info" dialog
- Render 3D window always brings the selected region's mean z position to 0 for easier visualization
- Render 3D: added buttons for xy, xz and yz projections
- Added DOIs related to G5M and axial loc. precision
- Removed mean frame filtering for G5M filtering/postprocessing
- Added tool tips to G5M dialog
- G5M automatically saves the check on relative sigma
- Updated Picasso Average documentation
- Changed default parameters in Simulate to reflect a typical DNA origami measurement
- SPINNA GUI allows for user-defined max y-axis value in the NND plot

### Bug fixes:

- Fixed 3D multichannel rendering
- Fixed Picasso Server launching in one-click-installers
- Fixed 3D MLE fitting and cleaned the docstrings for better readability (`picasso.gaussmle`)
- Fixed how Picasso: Simulates splits photons across binding events
- Fixed G5M 3D CI test
- Fixed Render 3D scale bar

## 0.9.7

### Important updates:

- Windows one-click-installer allows for selecting only a subset of Picasso modules to install
- Added ToRaw and Nanotron to one-click-installer
- *Experimental* One-click-installer for macOS (only for Apple Silicon), see [here](https://github.com/jungmannlab/picasso/tree/master/release/one_click_macos_gui)

### Small improvements:

- Adjusted the `config.yaml` and plugins instructions for the one-click-installer Picasso release (new Pyinstaller stores everything in the `_internal` folder)
- G5M output can save more columns (if present in the input localizations)
- Further enhancement of G5M documentation
- Render GUI: implemented filter by number of localizations for multichannel data
- Render GUI: allow removal of any column from localizations, not only `group`
- Filter GUI: allow removal of any column from localizations
- `REQUIRED_COLUMNS` moved from `picasso.localize` to `picasso.lib`

### Bug fixes:

- Fixed basic frame analysis in SMLM clusterer
- Fixed labels of the vertical lines in the subcluster test plot
- Fixed automatic Localize loading/unloading z-calibration paths when changing cameras
- Fixed `rel_sigma_z` in G5M (previously incorrectly divided by pixel size)
- Fixed G5M molmap `lpz` output
- Fixed loading square picks in Render
- Fixed appearance of the Apply expression dialog in Render for files with many columns
- Fixed initial x, y and N in LQ Gaussian fitting (might results in faster convergence and slightly different (<< NeNA) results) (#616)
- Fixed picking circular regions around left and top edges of the FOV

## 0.9.6

### Important updates:

- Test subclustering plot (saved after G5M, can be plotted in Filter): fixed the labels of the plots
- Change of API in `picasso.postprocess.nn_analysis`: new inputs cause backward compatibility issues. The function now returns only the nearest neighbor distances, not the indices of the nearest neighbors.

### Small improvements:

- Moved from merge sort to quick sort (usually faster due to lower memory usage)
- Render: increase the speed of picking circular locs, picking similar and filter by number of localizations (numba implementation)
- Render property histogram shown before rendering is activated
- Render property - removed legend
- Render Nearest Neighbor Analysis - saves nearest neighbors distances in the localizations .hdf5 file
- Render G5M dialog - adjusted the frame analysis checkbox
- Render G5M: removed the check for min. locs
- Render G5M: moved the check for too large clusters (or if any are present) before applying G5M to all channels (all channels analysis)
- Render masking: mask out saved area uses previously saved area if available in the metadata
- G5M documentation has been updated to include more troubleshooting tips and common issues, see [here](https://picassosr.readthedocs.io/en/latest/render.html#g5m)
- Localize zooms in and out centered at the current view
- Config file changes from 0.9.5 were [documented](https://picassosr.readthedocs.io/en/latest/localize.html) and [config template](https://github.com/jungmannlab/picasso/blob/master/picasso/config_template.yaml) was updated
- SPINNA 3D masking: z slicing added for visual inspection
- SPINNA 3D homogeneous simulations automatically adjusts the observed density based on the z range set by the user and the xy area of the pick (if provided)
- SPINNA allows for different mask bin size and blur in lateral and axial dimensions
- SPINNA default mask blur of 500 nm in the API (previously 65 nm)
- Reduced copying and conversion of DataFrames to numpy arrays (less memory usage)
- `picasso.io.load_locs` and `save_locs` ensure that the saved metadata contains the required keys
- Updated documentation on filetypes and minimum requirements for HDF5 files and accompanying YAML metadata files in Picasso
- Use `"col" in df.columns` instead of `hasattr(df, "col")` to check for columns in DataFrames (better readability)
- `picasso.postprocess` functions `picked_locs` and `pick_similar` accept precomputed index blocks to speed up the picking of circular regions
- One-click-installer's dependency on `pkg_resources` removed (since it has been removed from `setuptools`)
- Onc-click-installer: PyImarisWriter temporarily removed (caused problems with this release)

### Bug fixes:

- SPINNA 3D mask generation fixed (and `picasso.render.render_hist3d`)
- Test subcluster fix indexing
- Remove backward incompatible camera pixel size reading in SPINNA's mask generation (related to #602)
- Fixed localization masking for non-square mask (`picasso.masking.mask_locs`)
- Correct axial localization precision in Localize (magnification factor)
- Localize does not raise an error if QE is not found in the config file
- Localize does not automatically fit z coordinates if a 3D calibration file is loaded from the config file
- Render Test Clustering: fixed the full FOV button
- Fixed CLI `picasso join`

## 0.9.4-5

### Important updates:

- **Algorithm for molecular mapping introduced (G5M)**, see documentation [here](https://picassosr.readthedocs.io/en/latest/render.html#g5m). DOI: [10.1038/s41467-026-70198-5](https://doi.org/10.1038/s41467-026-70198-5)
- **Localize outputs axial localization precision for astigmatic imaging in 3D**. DOI: [10.1038/s41467-026-70198-5](https://doi.org/10.1038/s41467-026-70198-5)
- Localize GUI allows the user to select which localization columns to save when saving localizations. See the new dialog in the *File* -> *Select columns to save*
- Localize accepts frame bounds to analyze only a subset of frames
- Config file accepts z calibration .yaml paths so that they can be automatically loaded when changing between cameras
- Render by property (GUI) shows histogram of the selected property
- Filter GUI has a new plot to test for subclustering based on the number of events per molecule (column `n_events`); see the [Filter documentation](https://picassosr.readthedocs.io/en/latest/filter.html) for details

### *Small improvements:*

- Picasso applies constrained layout to all matplotlib figures
- SPINNA uses `FigureCanvas` instead of `QSvgRenderer` for displaying NND plots and mask legend
- SPINNA default mask blur set to 500 nm (GUI)
- 3D animation saves metadata
- Some improvements in how DataFrames are handled (Filter, change from `.values` to `.to_numpy()`)

### *Bug fixes:*

- Render GUI takes camera pixel size using `lib.get_from_metadata` (#602)
- Render by property is switched off if more than one channel is loaded
- Render 3D scale bar manual adjustment fixed
- Render 3D screenshot .yaml fixed
- .tif IO bug fix related to the numpy deprecation of `arr.newbyteorder` (#603)
- Clarify GPU fit installation instructions and remove version printing (#604)
- SPINNA fixed loading of the proportion spin boxes after rerunning SPINNA, such that they add up to 100% again

## 0.9.3

### Important updates:

- All GUI modules show the explanations of parameters when hovering over them with the mouse cursor (tool tips)
- FRC: does not blur rendered localizations, enabled saving rendered images
- Automatic testing at pull requests extended to most Picasso functions

### *Small improvements:*

- General improvements in the GUI widget names displayed (for example, change "Scalebar" to "Scale bar")
- Render: many input variables were switched from cam. pixels to nm in the GUI, for example, min. blur in the display settings dialog
- Render: slicer dialog automatically slices/unslices localizations when opening/closing the dialog
- Clustering algorithms copy the input localizations to avoid modifying the input DataFrame (for example, when using Picasso as a package)
- MLE Gauss fitting: default method is now `sigmaxy`, i.e., sigma can vary between x and y, like in the least-squares fitting
- Upgrade PyPI release action to release/v1 (security reasons)

### *Bug fixes:*

- Average: fix `pandas` warnings
- Localize: picasso.localize.identify accepts roi as input argument
- Render: show histogram in mask dialog ignores zero values
- Render: qPAINT histograms in the info dialog fixed and improved
- Fixed `picasso.postprocess.compute_local_density`

## 0.9.2

### Important updates:

- Improved and updated [sample notebooks](https://github.com/jungmannlab/picasso/tree/master/samples).
- Render: FRC resolution implementation, see DOI: [10.1038/nmeth.2448](https://doi.org/10.1038/nmeth.2448). It is calculated for a currently loaded FOV and only one repeat is done. *The exact implementation may change in the future versions.*

### *Small improvements:*

- `picasso.lib.get_from_metadata` function now has an option to raise a KeyError if the key is not found
- CMD: added undrift by fiducials (`picasso undrift_fiducials`)
- CMD: cleaned up .hdf5 conversion functions (`picasso hdf2csv`, `picasso csv2hdf` and [more](https://picassosr.readthedocs.io/en/latest/cmd.html))
- The above functions were moved to `picasso.io` module (previously only in `picasso.gui.render`)
- Picasso: Average CMD was removed since no functionality was implemented

### *Bug fixes:*

- AIM (`picasso.aim.aim`) copies localizations to avoid modifying the input DataFrame.
- AIM: fixed progress bar when no progress object is provided
- Localize: fixed CMD with GPUFit
- Simulate: fixed repetead axes tick labels
- SPINNA: fixed NND plot showing bins/lines outside of xlim
- SPINNA: extract the picked area based on the last .yaml file entry, not the first one (fixes the issue of incorrect densities extracted for localizations that were picked multiple times)
- SPINNA: enforce repeated generation of the search space when exp. data/densities/masks change
- CMD: pair correlation fixed (#588)

## 0.9.0-1

### Important updates:

- Picasso does not use `numpy.recarray` objects anymore. `pandas.DataFrame` are used instead. This applies to localizations, drift data, cluster centers, etc. **This change may cause backward compatibility issues when using Picasso as a package (downloaded from PyPI).**
- Updated other dependencies, most importantly, `numpy` is now in version 2
- Old setup files were replaced by `pyproject.toml` for building and packaging Picasso
- New option to save cluster areas/volumes in DBSCAN, HDBSCAN and SMLM clusterer using Otsu thresholding of rendered images
- Localize: ensure that 3D calibration is centered at z = 0; this guarantees the correct z scaling (magnification factor)
- Render: unfold groups was removed as it is contained within the square grid unfolding
- Render: new pick shape - square
- Render: synchronize groups across channels - removes localizations from groups that are not present in all channels, e.g., after filtering cluster centers by frame analysis, the cluster localizations corresponding to removed cluster centers are also removed
- Render: save pick properties extended to saving group properties, also qpaint index is saved
- SPINNA: improved saved fit results summary (see issue #560)

### *Small improvements:*

- Black-based code formatting applied to all scripts
- Cleaned up code for adjusting the size of QWidgets
- Progress dialog shows remaining time estimate more accurately (ignores the offset due to, for example, multiprocessing startup time)
- Render: save pick/group properties saves qpaint index (1 / mean dark time)
- Render: clustering metadata saves fraction of rejected localizations
- Render: screenshot .yaml files can be dragged and dropped to load the display settings
- Render: DBSCAN clustering .yaml file saves min. number of localizations per cluster
- Render 3D: display adjusted after changing blur method
- Localize: localization precision formula for least-squares fitting was corrected to account for a diagonal covariance Gaussian (background term is affected); the function for localization precision was moved from `picasso.postprocess` to `picasso.gausslq`
- SPINNA: GUI single sim does not allow the sum of proportions to exceed 100% (see issue #560)
- SPINNA: save last opened folder added
- SPINNA: smaller font size in NND plot for better readability
- SPINNA: clean up progress dialog
- SPINNA: NN plotting is normalized to 1000 nm
- Simplify the API for picking similar in `picasso.postprocess`

### *Bug fixes:*

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

## 0.8.8

- Render - masking dialog changed - threshold methods implemented, histogram of values shown, real-time rendering and different dialog layout
- Render - unfolding groups works without the Picasso: Average step beforehand
- Other bug fixes and minor improvements

## 0.8.5-7

- Sound notifications when long processes finish, see [here](https://picassosr.readthedocs.io/en/latest/others.html)
- Several dialogs in Render, Localize and Simulate are now scrollable (*experimental*)
- SPINNA fix automatic area detection from picked localizations
- Render add dependency `imageio[ffmpeg]` for building animations
- Render allow for loading pick regions by dropping a .yaml file onto the window
- Render improve zooming with mouse wheel (Ctrl/Cmd + wheel)
- Fast rendering automatically adjusts constrast
- Localize show scale bar function added
- Localize plotted ROI remains the same when zooming in/out and panning
- Localize Gauss MLE saves number of iterations and fit log-likelihood
- DBSCAN accepts min. no. of localizations per cluster
- Cluster center calculations calculate arithmetic mean, not weighted mean
- Other bug fixes and minor improvements

## 0.8.4

- SPINNA - easy fitting of labeling efficiency
- GUI docstrings added in all scripts; cleaned up docstrings in Picasso modules
- Render: pick size chosen in nm, not camera pixels
- Code clean up (flake8 compliant)
- Other bug fixes

## 0.8.3

- Design: fix export plates and pipetting schemes
- Design: set default biotin excess to 25 (previously set to 1)
- Render by property allows different colormaps
- Removed `lmfit` dependency
- Fix cluster centers bug from v0.8.2

## 0.8.2

- Added docstrings and data types in all modules (`postprocess`, `simulate`, `render`, `nanotron`, `localize`, `lib`, `io`, `imageprocess`, `gaussmle`, `gausslq`, `design`, `clusterer`, `aim`, `avgroi` and `zfit`)
- Fix one click installer issues for non-administrator users
- Render allows for saving picked localizations in a separate file for each pick
- Remaining time estimate in the progress dialog
- Fix garbage collection when openinging `.nd2` files in Localize
- Fix 3D rotation window for a polygon pick
- Render minimap - the zoom-in window is always visible
- Other small fixes and improvements

## 0.8.1

- Added `n_events` to cluster centers, i.e., number of binding events per cluster
- .yaml files contain Picasso version number for easier tracking
- Improved fiducial picking
- Bug fixes and other cosmetic changes

## 0.8.0

- **New module SPINNA for investigating oligormerization of proteins**, [DOI: 10.1038/s41467-025-59500-z](https://doi.org/10.1038/s41467-025-59500-z)
- **NeNA bug fix - old values were (usually) too high by a ~sqrt(2)**
- NeNA bug fix - less prone to fitting to local maximum leading to incorrect values
- NeNA plot - displays distances in nm
- Fiducial picking - filter out picks too few localizations (80% of the total acquisition time)
- `picasso csv2hdf` uses pandas to read .csv files
- Bug fixes

## 0.7.5

- Automatic picking of fiducials added in Render: `Tools/Pick fiducials`
- Undrifting from picked moved from `picasso/gui/render` to `picasso/postprocess`
- Plugin docs update
- Filter histogram display fixed for datasets with low variance (bug fix)
- AIM undrifting works now if the first frames of localizations are filtered out (bug fix)
- 2D drift plot in Render inverts y axis to match the rendered localizations
- 3D animation fixed
- Other minor bug fixes

## 0.7.1-4

- SMLM clusterer in picked regions deleted
- Show legend in Render property displayed rounded tick label values
- Pick circular area does not save the area for each pick in localization's metadata
- Picasso: Render - adjust the scale bar's size automatically based on the current FOV's width
- Picasso: Render - RESI dialog fixed, units in nm
- Picasso: Render - show drift in nm, not camera pixels
- Picasso: Render - masking localizations saves the mask area in its metadata
- Picasso: Render - export current view across channels in grayscale
- Picasso: Render - title bar displays the file only the names of the currently opened files
- CMD implementation of AIM undrifting, see `picasso aim -h` in terminal
- CMD localize saves camera information in the metadata file
- Other minor bug fixes

## 0.7.0

- Adaptive Intersection Maximization (AIM, doi: 10.1038/s41592-022-01307-0) implemented
- Z fitting improved by setting bounds on fitted z values to avoid NaNs
- CMD `clusterfile` fixed
- Picasso: Render 3D, rectangular and polygonal pick fixed
- `picasso.localize.localize` fixed
- default MLE fitting uses different sx and sy (CMD only)

## 0.6.9-11

- Added the option to draw polygon picks in Picasso: Render
- Save pick properties in Picasso: Render saves areas of picked regions in nm^2
- Calibration .yaml file saves number of frames and step size in nm
- `picasso.lib.merge_locs` function can merge localizations from multiple files
- Mask dialog in Picasso: Render saves .png mask files
- Mask dialog in Picasso: Render allows to save .png with the blurred image
- Picasso: Localize - added the option to save the current view as a .png file
- Picasso: Render - functions related to picking moved to `picasso.lib` and `picasso.postprocess`
- Picasso: Render - saving picked localizations saves the area(s) of the picked region(s) in the metadata file (.yaml)
- Documentation on readthedocs works again

## 0.6.6-8

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

## 0.6.3-5

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

## 0.6.2

- Picasso runs on Python 3.10 (jump from Python 3.7-3.8)
- New installation instructions
- Dependencies updated, meaning that M1 should have no problems with old versions of SciPy, etc.
- Localize: arbitrary number of sensitivity categories
- Picasso Render legend displays larger font
- Picasso Render Test Clusterer displays info when no clusters found instead of throwing an error
- Calling clustering functions from `picasso.clusterer` does not require camera pixel size. Same applies for the corresponding functions in CMD. *Only if 3D localizations are used, the pixel size must be provided.*
- HDBSCAN is installed by default since it is distributed within the new version of `scikit-learn 1.3.0`
- Screenshot `.yaml` file contains the list of colors used in the current rendering
- Render scale bar allows only integer values (i.e., no decimals)
- Localize .ims file fitting bug solve

## 0.6.1

- **Measuring in the 3D window (Measure and scale bar) fixed (previous versions did not convert the value correctly)**
- Localize GUI allows for numerical ROI input in the Parameters Dialog
- Allow loading individual .tif files as in Picasso v0.4.11
- RESI localizations have the new column `cluster_id`
- Building animation shows progress (Render 3D)
- Export current view in Render saves metadata; An extra image is saved with a scale bar if the user did not set it
- (**Not applicable in 0.6.2**) Clustering in command window requires camera pixel size to be input (instead of inserting one after calling the function)
- Bug fixes

## 0.6.0

- New RESI (Resolution Enhancement by Sequential Imaging) dialog in Picasso Render allowing for a substantial resolution boost, (*Reinhardt, et al., Nature, 2023.* DOI: 10.1038/s41586-023-05925-9)
- **Remove quantum efficiency when converting raw data into photons in Picasso Localize**
- Input ROI using command-line `picasso localize`, see [here](https://picassosr.readthedocs.io/en/latest/cmd.html).

## 0.5.7

- Updated installation instructions
- (H)DBSCAN available from cmd (bug fix)
- Render group information is faster (e.g., clustered data)
- Test Clusterer window (Render) has multiple updates, e.g., different projections, cluster centers display
- Cluster centers contain info about std in x,y and z
- If localization precision in z-axis is provided, it will be rendered when using `Individual localization precision` and `Individual localization precision (iso)`. **NOTE:** the column must be named `lpz` and have the same units as `lpx` and `lpy`.
- Number of CPU cores used in multiprocessing limited at 60
- Updated 3D rendering and clustering documentation
- Bug fixes

## 0.5.5-6

- Cluster info is saved in `_cluster_centers.hdf5` files which are created when `Save cluster centers` box is ticked
- Cluster centers contain info about group, mean frame (saved as `frame`), standard deviation frame, area/volume and convex hull
- `gist_rainbow` is used for rendering properties
- NeNA can be calculated many times
- Bug fixes

## 0.5.0-4

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
- [Plugins](https://picassosr.readthedocs.io/en/latest/plugins.html) can be added to all Picasso modules
- Many other improvements, bug fixes, etc.

## 0.4.6-11

- Logging for Watcher of Picasso Server
- Mode for multiple parameter groups for Watcher
- Fix for installation on Mac systems
- Various bugfixes

## 0.4.2-5

- Added more docstrings / documentation for Picasso Server
- Import and export for handling IMS (Imaris) files
- Fixed a bug where GPUFit was greyed out, added better installation instructions for GPUfit
- More documentation
- Added dockerfile

## 0.4.1

- Fixed a bug in installation

## 0.4.0

- Added new module "Picasso Server"
