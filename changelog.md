# Changelog

Last change: 08-SEP-2026 CEST

## Unreleased

### Localize
- New `picasso.localize.localize_frames`: a GUI-free wrapper that runs the existing identification and fit on an in-memory frame stack (instead of a movie read from disk) and assigns absolute frame indices, so batched or live input concatenates into one growing localization table. Results are numerically identical to `picasso.localize.localize` on the same frames and parameters.

## 0.11.1

### General
- Plugins can now extend Picasso's [Python API and command line](https://picassosr.readthedocs.io/en/latest/plugins.html#for-developers), not only the GUIs, and can be [installed and enabled without ever opening one](https://picassosr.readthedocs.io/en/latest/plugins.html#managing-plugins-without-a-gui) using the new [`picasso plugins`](https://picassosr.readthedocs.io/en/latest/cmd.html#plugins) command.
- Errors are no longer silently swallowed in the one-click installers. Picasso now redirects errors to a log file (`~/.picasso/logs/picasso.log`), logs every uncaught exception (main thread, worker threads and unraisable ones) there, and shows it in a message box - whose *Show Details...* holds the full traceback.

### Localize
- Localize can fit multichannel data one channel at a time, see the [documentation](https://picassosr.readthedocs.io/en/latest/localize.html#analyzing-each-channel-on-its-own).
- A lateral correction (astigmatism / chromatic) can now be kept in its own file and loaded separately from the 3D calibration, see the [documentation](https://picassosr.readthedocs.io/en/latest/localize.html#appending-or-loading-separately).
- New translation transform model (2 DOF, at least 1 bead pair), offered wherever Picasso fits a geometric transform. It fits a pure shift in x and y.
- Localize loads z-stack `.nd2` files.
- Localize's contrast is not set to auto on opening a new movie.
- ROI id is saved in single-channel localization (if ROIs are used).
- Fixed empty space in Localize's Parameters dialog.
- Fixed drawing ROIs in Localize while the temporal median filter is on.

### Render
- New pick shape in Render: **Box**, an axis-aligned rectangle dragged out to any size with the left mouse button. See the [documentation](https://picassosr.readthedocs.io/en/latest/render.html#picking-of-regions-of-interest).
- New pick shape in Render: **Brush**, painted freehand with the left mouse button. Strokes whose painted areas touch merge into one pick, a right click undoes the last stroke, and each stroke keeps the width it was painted with. See the [documentation](https://picassosr.readthedocs.io/en/latest/render.html#picking-of-regions-of-interest).
- Pick statistics (`View > Show info > Calculate info below`), `Filter picks by number of localizations` and `Select picks (XY scatter)` now work with every pick shape; they used to refuse anything but circles.
- Log-scale contrast sliders in Render: in the Display Settings dialogs of the main and of the 3D (rotation) window, and in the Test Clustering dialog.
- Render asks for the camera pixel size when loading `.hdf5` files whose metadata does not contain `Pixelsize` (e.g. saved by old Picasso versions), instead of raising an error at rendering.
- Render warns when the loaded channels have different camera pixel sizes and lets the user pick one pixel size for all of them, instead of raising an error at rendering.
- (Hopefully) fixed the problem of disappearing localizations when zoomed in.
- Fixed save cluster areas for SMLM clusterer ([#697](https://github.com/jungmannlab/picasso/issues/697)).
- Fixed the occasional "AttributeError: 'ViewRotation' object has no attribute 'viewport'" before the 3D window was opened.
- Fixed reading of old Picasso files ([#698](https://github.com/jungmannlab/picasso/pull/698)); thanks to @boydcpeters.
- Fixed `Undrift from picked` ignoring the pick shape: rectangular, square and polygonal picks were silently treated as circles (and polygons raised an error). `postprocess.undrift_from_fiducials` gained a `pick_shape` argument.
- Fixed right-clicking a polygon pick clearing every pick instead of the one clicked, and right-clicking anywhere deleting a perfectly vertical rectangular pick.
- Fixed saving rotated localizations (Render 3D) for polygonal picks.

### Others
- Fixed SPINNA simulations for empty targets.
- Average uses the shared global process pool: thus the initialization is not needed for each iteration.
- Capping multiprocessing at 61 processes only on Windows.
- More instructions on plugins.
- Clear error message in case opened localizations are not fully downloaded/copied.
- `io.load_picks` now raises a clear error for an unknown pick shape or a missing pick size, instead of `UnboundLocalError`/`KeyError`, and `render.draw_picks` raises instead of silently returning `None`.
- Templates for bug reports and feature/pull requests.
- Every GUI now starts through the shared `picasso.gui.app.run_gui`. Error reporting is installed *first*, so a failure while the main window is being built - a missing bundled library, a broken `settings.yaml` - is now shown and logged instead of killing the app without a word.
- Fixed progress dialog closing faster than within 0.5 s ([#700](https://github.com/jungmannlab/picasso/issues/700)).

## 0.11.0

This release substantially expands Picasso: Localize. Localization can now be performed with an experimentally measured PSF (cubic-spline model), jointly across several channels (e.g. biplane 3D), and with a pixel-dependent sCMOS noise model; rotated and spherical 2D Gaussian models were added as well. All GPU fitting was reimplemented in Numba CUDA, removing the dependency on Gpufit. Localize also reads a much wider range of data directly - `.tif` and OME-TIFF stacks (including movies split across several folders), MicroManager single-image acquisitions, Zeiss `.czi` and Leica `.lif` - so Picasso: ToRaw is no longer required and has been removed. Further additions include a temporal median filter for spot identification, lateral transform calibrations for astigmatism and chromatic aberration correction, localization metadata embedded in the `.hdf5` files, and a revised plugin system with an online plugin browser. We encourage all users to acquaint themselves with the new features in the [Localize documentation](https://picassosr.readthedocs.io/en/latest/localize.html). Render, SPINNA and the rest of the suite received numerous improvements as well, all listed below.

### **General important updates:**
- Localization metadata is now embedded directly in the `.hdf5` file (under the `/metadata` dataset, as a JSON string), making the file self-contained even if moved or renamed without its `.yaml` sidecar. When loading, Picasso looks for the metadata in the `.yaml` file first, then falls back to the embedded copy. Writing the `.yaml` sidecar is still on by default but can be disabled via the new user setting `Save metadata in .yaml` in `~/.picasso/settings.yaml` (also available via any module under File > Picasso settings). See [file formats documentation](https://picassosr.readthedocs.io/en/latest/files.html) for more info.
- Improved architecture for plugins, see [here](https://picassosr.readthedocs.io/en/latest/plugins.html). Note that the plugins should now be stored in a different location.
- Plugins can now be easily downloaded from our repository, using Plugins > Browse online plugins.
- `config.yaml` can now be stored in the `~/.picasso` directory and the location is easily accessible via Localize. `config.yaml` can still be read from the `picasso` folder for backward compatibility.
- Picasso: ToRaw was removed. Localize now reads multi-file OME-TIFF stacks (see below) and all other supported movie formats directly, so converting movies to `.raw` is no longer necessary. Existing `.raw` movies still load in Localize as before.
- PyQt6 is now imported lazily across the core library: the Qt widget classes formerly in `picasso.lib` moved to the new `picasso.lib_qt` module but remain accessible under their old `lib.<name>` names, and PyQt6 is only imported on first use — so `import picasso` and headless/CLI workflows no longer require PyQt6 to be installed.

#### Localize
- Picasso relies on package `tifffile` for processing `.tif` files and many other grayscale movie formats, see [localize documentation](https://picassosr.readthedocs.io/en/latest/localize.html). **Note:** this is an experimental feature, do not hesitate to let us know if you detect bugs/unexpected behavior or would like to see more file formats in Picasso, see our [GitHub page](https://github.com/jungmannlab/picasso/issues) for contact information.
- Added support for Zeiss `.czi` and Leica `.lif` movies in Localize (open dialog, drag-and-drop and batch CLI). These read via the optional `czifile` and `liffile` libraries (Python ≥ 3.12); install with `pip install picassosr[czi,lif]`. Multi-channel files prompt for a channel, and a `.lif` file with several acquisitions uses the one with the most frames.
- Added support for multichannel data, i.e., several movie files in a single Localize window. These can be analyzed sequentially or be treated as a multichannel data for combined localizations, for example, in biplane 3D imaging.
- Added support for MicroManager "separate image files" acquisitions (one `img_*.tif` per frame in a folder), see [Localize documentation](https://picassosr.readthedocs.io/en/latest/localize.html#extra-features).
- TIFF movies found across several folders can be opened as one concatenated movie (`File` > `Concatenate movies`), with the file order shown for confirmation before loading, see [Localize documentation](https://picassosr.readthedocs.io/en/latest/localize.html#extra-features).
- New fitting model: **Experimental PSF (cubic spline)** — fits an experimentally measured PSF (a cubic-spline model built from a bead z-stack), via the new `picasso.fitting.splinefit_cuda` module on the GPU or `picasso.fitting.splinefit` on the CPU. The spline coefficients of the calibration are computed in pure Python (NumPy/SciPy) on the CPU. In single-channel data, the bead alignment follows the workflow from [Li, et al, Nature Methods, 2018](https://www.nature.com/articles/nmeth.4661). See the [experimental PSF (cubic-spline) fitting documentation](https://picassosr.readthedocs.io/en/latest/localize.html#experimental-psf-cubic-spline-fitting) for details. *Note this is an experimental feature, do let us know if you find any bugs/unexpected behavior*.
- **Multichannel spline PSF fitting** (a shared-amplitude 3D spline model, e.g. biplane); additionally a new model was added for uncoupled photons with up to 6 channels. The global (multichannel) fitting follows globLoc, see [Li, et al, Nature Communications, 2022](https://doi.org/10.1038/s41467-022-30719-4).
- **Multichannel 2D spherical Gaussian fitting.** The spherical Gaussian can now be fitted jointly across several registered channels, sharing one position and width.
- **Standalone channel registration** (`Calibration` > `Register channels (2D)`, new module `picasso.registration`), which measures where each loaded channel sits relative to the first and saves it as its own small `.yaml`.
- New fitting algorithms supported: 2D rotated Gaussian, 2D spherical Gaussian.
- **GPU fitting is now implemented in Numba CUDA instead of Gpufit.** All seven models Picasso fits on the GPU — the spherical, elliptical and rotated 2D Gaussians and the cubic splines — now run through kernels written in Python and compiled at run time (`picasso.fitting.splinefit_cuda`, `picasso.fitting.gaussfit_cuda`, `picasso.fitting.lmfit_cuda`). The fitting algorithm itself is unchanged — it remains a port of [Gpufit](https://github.com/gpufit/Gpufit) (Przybylski et al., Scientific Reports 7, 15722, 2017). All CRLB calculations and parameter uncertainties are calculated in `picasso.fitting.precision`. Note that all the models are also available on the CPU.
- **sCMOS pixel-dependent noise model.** Picasso can now use a per-pixel camera calibration — offset, readout variance and, optionally, amplification gain — instead of the scalar `Baseline` and `Sensitivity`, and applies the noise model of [Huang et al., Nat. Methods 10, 653-658 (2013)](https://doi.org/10.1038/nmeth.2488) to MLE fitting. See the [Localize documentation](https://picassosr.readthedocs.io/en/latest/localize.html) for the acquisition protocol and the per-method behavior.
- **`picasso.gausslq` and `picasso.gaussmle` are deprecated and the whole modules will be removed in Picasso 1.0**, so that all fitting lives in the `picasso.fitting` subpackage. Every public name in them now raises a `DeprecationWarning` naming its replacement:
  - the fitters (`fit_spot`, `fit_spots`, `fit_spots_parallel`, `gaussmle`, `gaussmle_async`) → `picasso.fitting.gaussfit.fit_spots` / `fit_spots_async`.
  - `fit_spots_gauss_gpu` → `picasso.fitting.gaussfit_cuda.fit_spots`.
  - `locs_from_fits` → `picasso.localize.locs_from_fits_gauss`.
  - `localization_precision` and the two `sigma_uncertainty` functions → the new `picasso.fitting.precision` module, as `localization_precision`, `sigma_uncertainty_lsq` and `sigma_uncertainty_mle`.
  - Implementation of elliptical Gaussian MLE fitting on CPU changed slightly, now matching the results from Gpufit.
- New CPU fitting backend `picasso.fitting.gaussfit`: all three 2D Gaussian models (spherical, elliptical and rotated) over the same Levenberg-Marquardt driver as the GPU (Gpufit). Multithreading is used instead of multiprocessing.
- Z fitting (Gaussian-fitted localizations using astigmatism) on CUDA GPU.
- Gaussian filtering for spot identification of multiple-peak single-emitter images.
- New temporal median filter for spot identification with adaptable background, see [Martens, et al, Frontiers in Bioinformatics, 2022](https://doi.org/10.3389/fbinf.2021.817254). It is applied to the identification only (spots are always fitted on the raw movie) and it changes the scale of the net gradient, so `Min. net gradient` needs re-tuning when it is switched on or off.
- Lateral transform calibration for astigmatic imaging and chromatic abberation correction. A transform is fitted from two bead images (outlier bead pairs are rejected automatically) and appended to any calibration (Gaussian astigmatism `.yaml`, spline PSF `.hdf5`) or saved as a standalone calibration for 2D data; several corrections stack in one ordered list and are applied to `x`/`y` one after another (e.g. cylindrical lens, then chromatic) ([#15](https://github.com/jungmannlab/picasso/issues/15)) The transform model is selectable: **affine** (6 DOF, the default), **projective** (8 DOF, the perspective/keystone term a tilted dichroic introduces) or **polynomial2** / **polynomial3** (a smooth warp of that degree, following genuine field distortion).
- Accept multiple frame bounds.
- Accept multiple rectangular ROIs.
- Remove a ROI by double-clicking it in the preview.
- 3D calibration allows for multiple FOVs per z position (thanks to Aditya Ajay Chhatre for the suggestion).
- Movies now load on a background thread, so the Localize window stays responsive (and other windows are no longer blocked) while files are read; a progress dialog with a `Cancel` button is shown.
- Slider at the bottom of the app was added to allow easy navigation across frames.
- Contrast slider below it, with a black-point and a white-point handle.
- New keyboard shortcuts for navigating movies (move by 10/100/1,000 frames).
- Slight adjustments to some status bar messages.
- Cutting spots progress is reported between identification and fitting.
- Faster spot identification on network storage: `.tif`/`.ome.tif` and `.stk` movies are now read through a private file handle per worker thread instead of one shared, lock-serialized handle, so frame reads overlap and per-frame network latency is hidden.
- Faster spot cutting (`get_spots`) on network storage (see above).
- Smooth zooming in/out via scroll wheel.
- Contrast dialog's spin boxes use logarithmic scaling.
- Chi-square is saved for least-squares fitting results.
- GPU MLE-fitted localizations save log-likelihood and iterations.
- GPU Gaussian MLE-fitted localizations' precisions corrected/included (`lpx`, `photon_unc`, etc).
- Hovering over a fit marker or an identification box shows a tooltip listing the properties of the fitted localization (all saved columns, e.g. `x`, `y`, `photons`, `bg`) ([#239](https://github.com/jungmannlab/picasso/issues/239)).
- Loading a movie with corrupted metadata lets the user specify the most important ones without errors.
- Missing keys in MicroManager metadata based from camera configuration are ignored.
- Fixed ImageJ "contiguous stack" `.tif`/`.tiff` files (as written by ImageJ's "Save As > Tiff" for large stacks) being read as a single frame; all planes are now detected and read.
- Fixed a gap of roughly one box size in the identified spots along the borders between adjacent (e.g. overlapping) ROIs.
- Fixed aborting identification.
- Fixed reading of the movies from the network storage after interuption.

#### Render
- `Pick similar` now works with square and rectangular picks, not only circular ones, see [documentation](https://picassosr.readthedocs.io/en/latest/render.html#pick-similar-ctrl-shift-p).
- Rendering rotated Gaussians.
- More user-friendly measure tool.
- Faster AIM through smarter implementation.
- Localization files now load on a background thread, so the Render window (including the file dialogs) stays responsive while files are read; a progress dialog with a `Cancel` button is shown.
- Faster and more memory efficient (especially for large datasets) SMLM clusterer + progress bar.
- Progress bar for finding cluster centers.
- SMAP localization file reading, see "Other improvements" below.
- Rotation dialog allows for rotations around the localizations or the world (see [3D documentation](https://picassosr.readthedocs.io/en/latest/render.html#d-rotation-window)).
- Background color for multichannel data can be adjusted.
- Files dialog allows for better color selection of individual channels.
- Changes to the displayed channel selection (Files dialog) affect the 3D rotation window immediately.
- Right clicking a channel's checkbox in the Files dialog displays that channel only (all other channels are hidden).
- "Apply to all sequentially" available for drift correction algorithms, including drift from an external file.
- "Apply to all sequentially" available in Apply expression to localizations.
- G5M now supports 3D localizations fit with the experimental spline PSF, not only Gaussian fitting + astigmatism.
- G5M can model rotated (elliptical) molecules in 3D.
- G5M accepts `group_input` as the cluster id columns (useful if `group` is overwritten after clustering).
- Re-grouping localizations (picking, DBSCAN/HDBSCAN/SMLM clustering) now preserves the previous grouping in the `group_input` column instead of discarding it, so the original cluster ids remain available (e.g. for G5M).
- Updated G5M documentation - drift correction importance.
- Test clusterer with a constrast bar.
- Select localizations in the center of the binding event. See [Steen et al., Nature Methods 21, 1755-1762 (2024)](https://doi.org/10.1038/s41592-024-02374-8), Extended Data Fig. 1f.
- The picked regions can now be saved in the metadata when saving picked localizations, so the picks used to generate a file can be recovered from it. This is off by default and controlled by the new user setting `Save picks in metadata` in `~/.picasso/settings.yaml` (also available via any module under File > Picasso settings).
- Removed the switch to no blur method while panning.
- Improved robustness of NeNA calculation by trying out 3 starting positions for the fit.
- Fixed `picasso.g5m` bootstrap SEM (`bootstrap_check=True`) raising `AssertionError` for spline 3D data, because the resampling model was rebuilt without carrying over the fit mode.
- Fixed `picasso.g5m.sum_G5Ms` raising `TypeError` for a list of 2D models, because a calibration was passed to `G5M_2D`, which does not accept one.
- Fixed removed plugins menu after removing all localizations.
- Fixed pick similar numba error [#684](https://github.com/jungmannlab/picasso/issues/684).
- Fixed "Combine all channels" when saving localizations only saving the first channel when channels had differing columns.
- Fixed combined saving of multiple channels with different columns.
- Fixed ind. loc. precision rendering of `lpx = 0` [#685](https://github.com/jungmannlab/picasso/issues/685).
- Fixed z color rendering in 3D.
- Fixed `Length of values does not match length of index` error when saving pick properties in a channel where some picks contain no localizations.
- Fixed aspect ratio change by clicking Apply in the Change FOV dialog.
- Fixed the default directory in the load and save dialogs in the mask settings dialog.

#### SPINNA
- Adjustable font sizes and names for the NND plot's title, labels and ticks.
- Default NND plot title changed from "Nearest Neighbor Distances" to "NNDs".
- Fixed incorrect density of plotted molecules when using LE fitting.
- Fixed verbose for batch analysis.
- Batch analysis allows for specifying fitting mode (brute force/coarse to fine/bayesian).
- Batch analysis closes unused plots to save RAM.
- Removed the obsolete line of code in `_fill3d` ([#682](https://github.com/jungmannlab/picasso/issues/682)). This should not affect standard functionality of 3D masks; only the usage of `render_hist3d_anisotropic` directly might be affected.
- Fixed running a single simulation before fitting.

#### *Other improvements:*
- Added import and export of [SMAP](https://github.com/jries/SMAP) localizations (`_sml.mat`). Available in Render and as batch CLI converters `picasso smap2hdf` and `picasso hdf2smap`. Reads single-file MATLAB `-v7` and `-v7.3` saves.
- Removed folder `distribution` from the repository; `create_linux_shortcuts.py` was moved to `release`.
- Removed `notification_sounds` folder, the users can add their notification sounds in the `.picasso` folder.
- Progress reporting now goes through a uniform duck-typed interface (`lib.normalize_progress`): `None`, `"console"` and `lib.ProgressDialog` arguments are normalized once at the public entry points and driven with plain method calls, replacing the per-call-site `isinstance`/`"console"` branches — so headless runs never touch Qt at runtime either. `lib.TqdmProgress` and `lib.MockProgress` now implement the full `ProgressDialog` interface (`setMaximum`, `maximum`, `zero_progress`, `close`).
- Localizations imported from ThunderSTORM and SMAP maintain all their columns, not only the Picasso pre-defined ones.
- `Micro-Manager Metadata` block (the microscope properties read from a MicroManager movie) can now be left out when localizing, see [documentation](https://picassosr.readthedocs.io/en/latest/files.html#metadata).
- `Micro-Manager Acquisition Comments` is only saved in the metadata if the acquisition actually has a comment; an empty one is no longer written.
- All API docstrings have been updated to match the [Numpy docstrings style](https://numpydoc.readthedocs.io/en/latest/format.html).
- Menu entries that open a dialog end with an ellipsis (e.g. `File > Open...`), following the standard GUI convention.
- Closing the Localize window while an identification or a fit is running now stops that worker first.

### **Backward incompatible changes:**
- All the functions deprecated in v0.10 were removed, see section **0.10.0** below.
- `picasso.clusterer.cluster_center` removed (the functionality provided by `find_cluster_centers`).
- `render.render` only keyword arguments except `locs` and `info`.
- Nanotron accepts `disp_px_size` instead of `oversampling` for easier user interaction.
- Plugins location changed, see [here](https://picassosr.readthedocs.io/en/latest/plugins.html); `picasso/gui/plugins` folder removed.
- `picasso.gui.toraw`, the CLI command `picasso toraw` and the conversion functions `picasso.io.to_raw`, `picasso.io.to_raw_combined` and `picasso.io.get_movie_groups` were removed.
- `picasso.postprocess.pick_similar` takes `pick_shape` and `pick_size` instead of `d`, matching `picked_locs`. `pick_size` keeps the usual meaning per shape (diameter for circles, side length for squares, width for rectangles), so `pick_similar(locs, info, picks, d=1.5)` becomes `pick_similar(locs, info, picks, "Circle", 1.5)`.

#### *Deprecation warnings:*
- **`picasso.gausslq` and `picasso.gaussmle` are deprecated and will be removed in Picasso 1.0.**.
- `picasso.localize.identify` and `picasso.localize.localize` will always return metadata in v0.12.0, `return_info` will no longer be accepted.
- `picasso.localize.fit2D` was renamed to `picasso.localize.fit` and will be removed in v0.12.0, together with its `movie_info` and `mle_method` arguments, neither of which affects the fit.
- `picasso.localize.localize_3D` will be removed in v0.12.0; `picasso.localize.localize` now takes the astigmatism calibration as `calibration_3d` and fits z itself, which was the only difference between the two functions.
- `picasso.localize.localize` will only accept the movie as a positional argument in v0.12.0; `camera_info` and `identification_parameters` become keyword-only.
- `picasso.localize.localize`'s `parameters` argument was renamed to `identification_parameters` and now also carries the spatial Gaussian filter (as the `Gaussian Filter Sigma` key, replacing the `gaussian_filter_sigma` argument added earlier in this release); `parameters` will be removed in v0.12.0.
- `picasso.localize.localize`'s `mle_method` argument is ignored and will be removed in v0.12.0.
- `eps` and `max_it` are honored by every iterating fitting method on either device (all of them except `avg`).

## 0.10.3
- Fixed total pick area in the .yamls for circular and square picks (Render).
- Fixed plotting x and y in "Select picks (trace) in Render".

## 0.10.2
- Rotations use quaterions for unambiguous workflow, fixing bugs [#673](https://github.com/jungmannlab/picasso/issues/673), [#674](https://github.com/jungmannlab/picasso/issues/674) and [#675](https://github.com/jungmannlab/picasso/issues/675).
- All 3 angles in "Rotate by angle" in the 3D rotation window are accumulated into a single widget.
- Render by property fixed for large files ([#677](https://github.com/jungmannlab/picasso/issues/677)), possibly related to [#672](https://github.com/jungmannlab/picasso/issues/672).
- Fixed "Best fitting combination" button in SPINNA ([#676](https://github.com/jungmannlab/picasso/issues/676)).
- Fixed 3D calibration when frame range is user-specified.
- Fixed redoing 3D calibration when identification parameters change.

## 0.10.1

#### Localize
- Fixed Gauss-fitting error when spot's sum is zero (zero division error).

#### Render
- Faster rendering through improvements for all blur methods and multi-level spatial indexing for quick zoomed-in rendering.
- Multichannel rendering supports colormaps, not only a single RGB color.
- Anisotripic DBSCAN (with faster implementation) [DOI: 10.1021/acs.jpcb.4c02030](https://doi.org/10.1021/acs.jpcb.4c02030).
- Faster cluster centers calculation for SMLM clusterer, DBSCAN and HDBSCAN + `lpz` is saved if applicable.
- Load FOV keeps the aspect ratio of the input .txt file.
- Files dialog resizing fixed.
- Show 3D clustering widgets only when 3D data is loaded.
- Fixed linking saving `lpz`.
- Fixed high-resolution display of mask in Render (#666).
- Removed render property cache since it did not provide any significant speed improvement.

#### SPINNA
- Comparing models uses fitting modes (Bayesian, etc) and has a cleaner progress dialog.
- Convenient fitting of LE.
- Area/volume button removed (deduced automatically from densities and number of molecules in the exp. data).
- Show 3D masking widgets only if 3D mask is selected.
- Batch analysis for LE fitting.
- Batch analysis does not require area input (if found in metadata).
- Batch analysis has clear instructions on what columns are required via CLI and the docs update.
- Fixed NND plot reindexing after fitting (#665).

#### Filter
- Efficient Filter: much lower RAM usage + faster filtering by histogram selection (1D/2D), especially for very large datasets.

#### Others
- Expanded the scope of the sample notebooks.
- Improved docstrings for 3D SMLM clusterer.
- Flake8 clean-up.
- Installers are distributed with readme.txt files (previously .rst).
- Minor changes to documentation.
- Fixed subcluster check plot when one of the two populations is empty (#667).
- Fixed 2D fitting with console printout and no multiprocessing.

## 0.10.0

### **General updates:**

- Numerous new functions added in the API to simplify the more complicated analyses, for example, `picasso.localize.fit2D`.
- Installing Picasso as a package has less stringent dependencies and Python version requirements, the exact versions are specified for one-click-installers only.
- Almost all the functions in the GUI scripts (for example, `picasso.gui.render.py`) not related to GUI were moved to corresponding API scripts such that using Picasso as a Python package allows for easy analysis analogous to what GUI provides. For example, ``picasso.render.py`` does not only provide the function to generate a grayscale image of localizations only (like before) but can also be used to paint the same images with a color map as they are rendered (for example, with picks and scale bar).
- One-click installer uses Python 3.14 (previously 3.10) and updated dependencies, which should improve the performance of some functions.
- Easy access to user settings via any Picasso module.
- Expanded test suite (CI).
- Picasso automatically checks for updates when launched and notifies the user if a new version is available.
- Render, Localize, Average and Filter allow the user to inspect metadata in the app.

#### Localize
- [GPUfit](https://github.com/gpufit/Gpufit) incorporated into Picasso (`picasso.ext.pygpufit`).
- Localize supports .stk file format from MetaMorph (*experimental*).
- Abort button to stop asynchronous multiprocessing (for example, during identification).
- Error box compatible with multiprocessed tasks (clear error message).
- Save and load identifications.
- Save spots as .tif, .npy, not .hdf5.
- Documentation updated relating to the file menu features, such as loading picks as identifications.
- Fixed reading .ims movies.
- Fixed spot saving.
- Export current view is less pixelated.
- Localization markers (green crosses) in the GUI are not affected by drift correction (only visual improvement).
- CLI `picasso localize <files>` allows for MLE fitting in 3D (z-fitting still as per Huang et al, 2008.).

#### Render
- Smarter fast rendering, lowering RAM usage almost two-fold.
- Faster ind. loc. precision rendering in 3D.
- Test clustering supports G5M.
- Test clustering saves the channel to which the algorithms are applied.
- Test clustering allows for applying the current parameters to the whole dataset.
- Test clustering tool tips.
- G5M calculates more accurate sigma constraints in 3D.
- Plot localizations profile for rectangular pick.
- Reading .csv files from ThunderSTORM.
- Mask settings dialog allows for zooming and panning.
- More accessible saving/loading of FOVs as .txt files.
- Show NeNA/FRC plot buttons automatically calculate them if not done already.
- Keyboard shortcut for closing all localizations (Ctrl+Shift+Backspace or Ctrl+Shift+Delete).
- Legend is displayed on black background for better visibility.
- Log-scaling of contrast.
- New image exporting with manually selected rendering options + support for .pdf and .svg formats.
- Optimal scale bar is only set upon user's request.
- Changed the name "Nearest Neighbor Analysis" to "Calculate nearest neighbor distances" for better clarity.
- Faster non-circle picking by smarter indexing.
- Trace shows number of photons in addition to x, y and frame; exports .csv files with three columns (frame, ON/OFF and photons).
- Manual setting of scale bar switches off automatic scale bar length.
- Apply drift from external file supports dropping the .txt file onto the window.
- Show drift keeps x and y coords to scale (on the second plot).
- 3D rotation window supports rendering by property.
- More intuitive rotation in 3D instead of simple rotations around xyz axes.
- Animation dialog allows unlimited positions.
- Fixed 3D animation for non-square FOV.
- Fixed distances in NeNA plot (previously plotting multiple times kept increasing the values).
- Fixed panning in 3D.
- Fixed 3D screenshot metadata.
- Fixed pre-G5M group/max locs checks when applying to all channels.
- Fixed zero-value in rendered images (previously RGB channels were capped between 1 and 255 instead of 0 and 255).
- Fixed default directory for applying drift from external file.
- Added attribute `pixelsize` in View for cleaner code.

#### SPINNA
- Two new fitting methods for fast fitting instead of the brute force search, see [documentation](https://picassosr.readthedocs.io/en/latest/spinna.html#fitting).
- User-defined threshold for the binary mask.
- Loading new structures in the Simulate tab without changing targets does not reset the window.
- Fixed .svg saving in the one-click-installer app.
- Fixed issues caused by removing structures in the Structures Tab (Windows).

#### Filter
- Support for .csv export (not only hdf5).
- Apply filtering steps from metadata.
- Filtering range for numerical filtering is inclusive.

#### Average
- Abort button.
- Improved saved metadata.
- Adjusted default parameters.

#### *Other improvements:*
- Only `picasso.version.py` determines software version globally, thus `bumpversion` is not needed anymore.
- `picasso.lib.merge_locs` allows for flexible `frame` and `group` incrementing when merging localizations lists.
- New functions in the API `picasso.postprocess.undrift_from_fiducials` and `picasso.postprocess.apply_drift` that can be used to undrift localizations based on picked fiducials with or without user-specified picks and to apply the calculated drift to the localizations, respectively.
- New API for alignement of locs, see ``picasso.postprocess``: ``align_rcc`` and ``align_from_picked``.
- New function ``picasso.io.load_picks``.
- Adjusted installation instructions in README.
- Badges added to the GitHub repository (PyPI and Python versions, changelog).
- Dialogs with scroll areas show no margins (e.g., Display settings dialog in Render).
- Added help buttons to some dialogs/menu bars across the modules that open the corresponding readthedocs pages (the documentation will be further improved in the future).
- "What's this?" help button removed from all dialogs (Windows) as it previously crashed Picasso.
- Changelog changed from .rst to markdown for GitHub display.
- Removed focus on push buttons in dialogs.
- Improved data typing of numpy arrays.
- Fixed flake8 warnings (code style only).
- `picasso.postprocess.groupprops` shows no progress by default.
- `picasso.io.TiffMultiMap` docstrings corrected.
- CLI function `nneighbor` uses KDTree for higher speed.
- Picasso: Simulate (multilabel) saves label names as in "Exchange rounds to be simulated" rather than 0, 1, 2, ...
- Fixed Picasso: ToRaw.
- `path.replace()` is no longer used to change the extension of the path (safer approach).

### **Backward incompatible changes:**

- Several new depedencies have been added. If Picasso is installed via PyPI (`pip install picassosr`) or one-click-installer, no action needs to be taken. **Otherwise please install them when updating Picasso to v0.10.0**. The dependencies are: `tifffile`, `hdf5plugin` (only for Windows to read .ims files). Additionally `PyQt5` was updated to `PyQt6`.
- `picasso.spinna.SPINNA.fit` accepts all inputs as keyword arguments (except for `N_structures`).
- Names of nearly all functions in `picasso.g5m` and some in `picasso.zfit` have been changed (underscore added to prefix as private functions). The main functions in these scripts were left unchanged: `g5m.g5m`, `zfit.zfit`. Functions `zfit.fit_z` and `zfit.fit_z_parallel` are deprecated, see below.
- Cluster centers (DBSCAN, HDBSCAN, SMLM clusterer) save number of localizations per cluster as `n_locs`, not `n`.

#### *Deprecation warnings:*

- `picasso.lib.unpack_calibration` and the `spot_size`, `z_range` parameters in the G5M functions. `picasso.g5m.g5m` now uses calibration coefficients only for setting sigma constraints in 3D for more accurate results.
- `picasso.clusterer.cluster_center` (will be renamed to `_cluster_center` and become a private function in v0.11.0).
- `picasso.aim`: `intersect1d`, `count_intersections`, `run_intersections`, `run_intersections_multithread`, `get_fft_peak`, `get_fft_peak_z`, `point_intersect_2d` and `point_intersect_3d` (will become private functions in v0.11.0).
- `picasso.masking.mask_locs` uses metadata rather than now deprecated `width` and `height` parameters.
- `picasso.spinna.MaskGenerator`: `run_checks` parameter (will be removed in v0.11.0).
- `picasso.localize.identify` and `picasso.localize.localize` will return metadata by default in v0.11.0.
- `fit_z` and `fit_z_parallel` in `picasso.zfit` will be deprecated in v0.11.0. `zfit.zfit` takes over as the main function in the script.
- `picasso.render` takes in `disp_px_size` rather than `oversampling`, see the function; `oversampling` will be removed in v0.11.0.
- `picasso.render` functions: `render_hist`, `render_gaussian`, `render_gaussian_iso`, `render_smooth` and `render_convolve` will become private in v0.11.0.
- `picasso.gausslq.initial_parameters_gpufit` and `picasso.gaussmle.mean_filter` will become private in v0.11.0.
- `picasso.localize` functions: `local_maxima`, `gradient_at`, `net_gradient` will become private in v0.11.0. Functions `fit` and `fit_async` will be removed entirely.
- `picasso.postprocess` functions: `index_blocks_shape`, `n_block_locs_at`, `next_frame_neighbor_distance_histogram`, `get_link_groups` and `link_loc_groups` will become private in v0.11.0.
- `picasso.spinna` functions: `find_target_counts`, `get_structures_permutation`, `targets_from_structures`.

## 0.9.10

### Important updates:

- Added support for loading BigTIFF in Picasso Localize (#631), big thanks to @boydcpeters.

### Small improvements:

- `picasso.aim.aim` accepts progress as a `lib.ProgressDialog`, `"console"` or `None`.
- SPINNA GUI: Small adjustment to GUI when loading search space.
- Adjusted label in subcluster check plot.
- Subcluster check plot outputs p value and test statistic.

### Bug fixes:

- Fixed AIM in Localize GUI.
- Fixed saving search space in SPINNA for multiple-target structures.

## 0.9.8-9

### Small improvements:

- Added a function `picasso.lib.get_save_filename_ext_dialog` that can also check for the existence of the files with other extenstions (for example, if the user tries to save a .yaml file with the same name as an existing .hdf5 file, it will ask if the user wants to overwrite the .hdf5 file). This is implemented in all GUI modules when saving files.
- `PyImarisWriter` is included in the one-click-installer again (Windows only).
- Localize GUI allows the user to automatically undrift localizations.
- Localize Parameters dialog displays a message if the z calibration path in the config file could not be found.
- MLE fitting saves CRLB uncertainties of fitted parameters: photons, background, sx and sy.
- `picasso.localize.fit` default method changed to `sigmaxy` (anisotropic sigma fitting).
- Render export localizations supports exporting all channels sequentially.
- Changed default max. frames in linking (dark times calculation) to 3 (previously 1) (both GUI and `picasso.postprocess.link`).
- Added number of binding events to Render's "Show info" dialog.
- Render 3D window always brings the selected region's mean z position to 0 for easier visualization.
- Render 3D: added buttons for xy, xz and yz projections.
- Added DOIs related to G5M and axial loc. precision.
- Removed mean frame filtering for G5M filtering/postprocessing.
- Added tool tips to G5M dialog.
- G5M automatically saves the check on relative sigma.
- Updated Picasso Average documentation.
- Changed default parameters in Simulate to reflect a typical DNA origami measurement.
- SPINNA GUI allows for user-defined max y-axis value in the NND plot.

### Bug fixes:

- Fixed 3D multichannel rendering.
- Fixed Picasso Server launching in one-click-installers.
- Fixed 3D MLE fitting and cleaned the docstrings for better readability (`picasso.gaussmle`).
- Fixed how Picasso: Simulates splits photons across binding events.
- Fixed G5M 3D CI test.
- Fixed Render 3D scale bar.

## 0.9.7

### Important updates:

- Windows one-click-installer allows for selecting only a subset of Picasso modules to install.
- Added ToRaw and Nanotron to one-click-installer.
- *Experimental* One-click-installer for macOS (only for Apple Silicon), see [here](https://github.com/jungmannlab/picasso/tree/master/release/one_click_macos_gui).

### Small improvements:

- Adjusted the `config.yaml` and plugins instructions for the one-click-installer Picasso release (new Pyinstaller stores everything in the `_internal` folder).
- G5M output can save more columns (if present in the input localizations).
- Further enhancement of G5M documentation.
- Render GUI: implemented filter by number of localizations for multichannel data.
- Render GUI: allow removal of any column from localizations, not only `group`.
- Filter GUI: allow removal of any column from localizations.
- `REQUIRED_COLUMNS` moved from `picasso.localize` to `picasso.lib`.

### Bug fixes:

- Fixed basic frame analysis in SMLM clusterer.
- Fixed labels of the vertical lines in the subcluster test plot.
- Fixed automatic Localize loading/unloading z-calibration paths when changing cameras.
- Fixed `rel_sigma_z` in G5M (previously incorrectly divided by pixel size).
- Fixed G5M molmap `lpz` output.
- Fixed loading square picks in Render.
- Fixed appearance of the Apply expression dialog in Render for files with many columns.
- Fixed initial x, y and N in LQ Gaussian fitting (might results in faster convergence and slightly different (<< NeNA) results) (#616).
- Fixed picking circular regions around left and top edges of the FOV.

## 0.9.6

### Important updates:

- Test subclustering plot (saved after G5M, can be plotted in Filter): fixed the labels of the plots.
- Change of API in `picasso.postprocess.nn_analysis`: new inputs cause backward compatibility issues. The function now returns only the nearest neighbor distances, not the indices of the nearest neighbors.

### Small improvements:

- Moved from merge sort to quick sort (usually faster due to lower memory usage).
- Render: increase the speed of picking circular locs, picking similar and filter by number of localizations (numba implementation).
- Render property histogram shown before rendering is activated.
- Render property - removed legend.
- Render Nearest Neighbor Analysis - saves nearest neighbors distances in the localizations .hdf5 file.
- Render G5M dialog - adjusted the frame analysis checkbox.
- Render G5M: removed the check for min. locs.
- Render G5M: moved the check for too large clusters (or if any are present) before applying G5M to all channels (all channels analysis).
- Render masking: mask out saved area uses previously saved area if available in the metadata.
- G5M documentation has been updated to include more troubleshooting tips and common issues, see [here](https://picassosr.readthedocs.io/en/latest/render.html#g5m).
- Localize zooms in and out centered at the current view.
- Config file changes from 0.9.5 were [documented](https://picassosr.readthedocs.io/en/latest/localize.html) and [config template](https://github.com/jungmannlab/picasso/blob/master/picasso/config_template.yaml) was updated.
- SPINNA 3D masking: z slicing added for visual inspection.
- SPINNA 3D homogeneous simulations automatically adjusts the observed density based on the z range set by the user and the xy area of the pick (if provided).
- SPINNA allows for different mask bin size and blur in lateral and axial dimensions.
- SPINNA default mask blur of 500 nm in the API (previously 65 nm).
- Reduced copying and conversion of DataFrames to numpy arrays (less memory usage).
- `picasso.io.load_locs` and `save_locs` ensure that the saved metadata contains the required keys.
- Updated documentation on filetypes and minimum requirements for HDF5 files and accompanying YAML metadata files in Picasso.
- Use `"col" in df.columns` instead of `hasattr(df, "col")` to check for columns in DataFrames (better readability).
- `picasso.postprocess` functions `picked_locs` and `pick_similar` accept precomputed index blocks to speed up the picking of circular regions.
- One-click-installer's dependency on `pkg_resources` removed (since it has been removed from `setuptools`).
- Onc-click-installer: PyImarisWriter temporarily removed (caused problems with this release).

### Bug fixes:

- SPINNA 3D mask generation fixed (and `picasso.render.render_hist3d`).
- Test subcluster fix indexing.
- Remove backward incompatible camera pixel size reading in SPINNA's mask generation (related to #602).
- Fixed localization masking for non-square mask (`picasso.masking.mask_locs`).
- Correct axial localization precision in Localize (magnification factor).
- Localize does not raise an error if QE is not found in the config file.
- Localize does not automatically fit z coordinates if a 3D calibration file is loaded from the config file.
- Render Test Clustering: fixed the full FOV button.
- Fixed CLI `picasso join`.

## 0.9.4-5

### Important updates:

- **Algorithm for molecular mapping introduced (G5M)**, see documentation [here](https://picassosr.readthedocs.io/en/latest/render.html#g5m). DOI: [10.1038/s41467-026-70198-5](https://doi.org/10.1038/s41467-026-70198-5).
- **Localize outputs axial localization precision for astigmatic imaging in 3D**. DOI: [10.1038/s41467-026-70198-5](https://doi.org/10.1038/s41467-026-70198-5).
- Localize GUI allows the user to select which localization columns to save when saving localizations. See the new dialog in the *File* -> *Select columns to save*.
- Localize accepts frame bounds to analyze only a subset of frames.
- Config file accepts z calibration .yaml paths so that they can be automatically loaded when changing between cameras.
- Render by property (GUI) shows histogram of the selected property.
- Filter GUI has a new plot to test for subclustering based on the number of events per molecule (column `n_events`); see the [Filter documentation](https://picassosr.readthedocs.io/en/latest/filter.html) for details.

### *Small improvements:*

- Picasso applies constrained layout to all matplotlib figures.
- SPINNA uses `FigureCanvas` instead of `QSvgRenderer` for displaying NND plots and mask legend.
- SPINNA default mask blur set to 500 nm (GUI).
- 3D animation saves metadata.
- Some improvements in how DataFrames are handled (Filter, change from `.values` to `.to_numpy()`).

### *Bug fixes:*

- Render GUI takes camera pixel size using `lib.get_from_metadata` (#602).
- Render by property is switched off if more than one channel is loaded.
- Render 3D scale bar manual adjustment fixed.
- Render 3D screenshot .yaml fixed.
- .tif IO bug fix related to the numpy deprecation of `arr.newbyteorder` (#603).
- Clarify GPU fit installation instructions and remove version printing (#604).
- SPINNA fixed loading of the proportion spin boxes after rerunning SPINNA, such that they add up to 100% again.

## 0.9.3

### Important updates:

- All GUI modules show the explanations of parameters when hovering over them with the mouse cursor (tool tips).
- FRC: does not blur rendered localizations, enabled saving rendered images.
- Automatic testing at pull requests extended to most Picasso functions.

### *Small improvements:*

- General improvements in the GUI widget names displayed (for example, change "Scalebar" to "Scale bar").
- Render: many input variables were switched from cam. pixels to nm in the GUI, for example, min. blur in the display settings dialog.
- Render: slicer dialog automatically slices/unslices localizations when opening/closing the dialog.
- Clustering algorithms copy the input localizations to avoid modifying the input DataFrame (for example, when using Picasso as a package).
- MLE Gauss fitting: default method is now `sigmaxy`, i.e., sigma can vary between x and y, like in the least-squares fitting.
- Upgrade PyPI release action to release/v1 (security reasons).

### *Bug fixes:*

- Average: fix `pandas` warnings.
- Localize: picasso.localize.identify accepts roi as input argument.
- Render: show histogram in mask dialog ignores zero values.
- Render: qPAINT histograms in the info dialog fixed and improved.
- Fixed `picasso.postprocess.compute_local_density`.

## 0.9.2

### Important updates:

- Improved and updated [sample notebooks](https://github.com/jungmannlab/picasso/tree/master/samples).
- Render: FRC resolution implementation, see DOI: [10.1038/nmeth.2448](https://doi.org/10.1038/nmeth.2448). It is calculated for a currently loaded FOV and only one repeat is done. *The exact implementation may change in the future versions.*.

### *Small improvements:*

- `picasso.lib.get_from_metadata` function now has an option to raise a KeyError if the key is not found.
- CMD: added undrift by fiducials (`picasso undrift_fiducials`).
- CMD: cleaned up .hdf5 conversion functions (`picasso hdf2csv`, `picasso csv2hdf` and [more](https://picassosr.readthedocs.io/en/latest/cmd.html)).
- The above functions were moved to `picasso.io` module (previously only in `picasso.gui.render`).
- Picasso: Average CMD was removed since no functionality was implemented.

### *Bug fixes:*

- AIM (`picasso.aim.aim`) copies localizations to avoid modifying the input DataFrame.
- AIM: fixed progress bar when no progress object is provided.
- Localize: fixed CMD with GPUFit.
- Simulate: fixed repetead axes tick labels.
- SPINNA: fixed NND plot showing bins/lines outside of xlim.
- SPINNA: extract the picked area based on the last .yaml file entry, not the first one (fixes the issue of incorrect densities extracted for localizations that were picked multiple times).
- SPINNA: enforce repeated generation of the search space when exp. data/densities/masks change.
- CMD: pair correlation fixed (#588).

## 0.9.0-1

### Important updates:

- Picasso does not use `numpy.recarray` objects anymore. `pandas.DataFrame` are used instead. This applies to localizations, drift data, cluster centers, etc. **This change may cause backward compatibility issues when using Picasso as a package (downloaded from PyPI).**.
- Updated other dependencies, most importantly, `numpy` is now in version 2.
- Old setup files were replaced by `pyproject.toml` for building and packaging Picasso.
- New option to save cluster areas/volumes in DBSCAN, HDBSCAN and SMLM clusterer using Otsu thresholding of rendered images.
- Localize: ensure that 3D calibration is centered at z = 0; this guarantees the correct z scaling (magnification factor).
- Render: unfold groups was removed as it is contained within the square grid unfolding.
- Render: new pick shape - square.
- Render: synchronize groups across channels - removes localizations from groups that are not present in all channels, e.g., after filtering cluster centers by frame analysis, the cluster localizations corresponding to removed cluster centers are also removed.
- Render: save pick properties extended to saving group properties, also qpaint index is saved.
- SPINNA: improved saved fit results summary (see issue #560).

### *Small improvements:*

- Black-based code formatting applied to all scripts.
- Cleaned up code for adjusting the size of QWidgets.
- Progress dialog shows remaining time estimate more accurately (ignores the offset due to, for example, multiprocessing startup time).
- Render: save pick/group properties saves qpaint index (1 / mean dark time).
- Render: clustering metadata saves fraction of rejected localizations.
- Render: screenshot .yaml files can be dragged and dropped to load the display settings.
- Render: DBSCAN clustering .yaml file saves min. number of localizations per cluster.
- Render 3D: display adjusted after changing blur method.
- Localize: localization precision formula for least-squares fitting was corrected to account for a diagonal covariance Gaussian (background term is affected); the function for localization precision was moved from `picasso.postprocess` to `picasso.gausslq`.
- SPINNA: GUI single sim does not allow the sum of proportions to exceed 100% (see issue #560).
- SPINNA: save last opened folder added.
- SPINNA: smaller font size in NND plot for better readability.
- SPINNA: clean up progress dialog.
- SPINNA: NN plotting is normalized to 1000 nm.
- Simplify the API for picking similar in `picasso.postprocess`.

### *Bug fixes:*

- Render: unfold groups/picks (rectangular grid) fixed for nonconsecutive grouping (the grid might have had missing elements before).
- Render: apply drift from external file fixed.
- Render: fix masking (issue #560).
- Render: fix loading camera pixel size from metadata (see issue #560).
- Render: saving picks separately fixed areas in the .yaml files.
- Render: loading a new channel with rendering by property fixed.
- Render: mouse events are ignored if no localizations are loaded.
- Render 3D: remove measurement points fixed.
- Render 3D: save rotated localizations fixed.
- Render 3D: fixed ind. loc. prec.
- Render 3D: rendering an empty pick fixed.
- Localize: user-friendly display of large numbers (for example, 1,052,102 instead of 1052102).
- Localize: fixed acquisition comment extraction from uManager .tif files.
- SPINNA: fixed all the bugs related to masking and search space generation (see issue #560).
- SPINNA: save NND plot fixed (when no simulations were run).
- SPINNA: read camera pixel size from metadata fixed (if available).

## 0.8.8

- Render - masking dialog changed - threshold methods implemented, histogram of values shown, real-time rendering and different dialog layout.
- Render - unfolding groups works without the Picasso: Average step beforehand.
- Other bug fixes and minor improvements.

## 0.8.5-7

- Sound notifications when long processes finish, see [here](https://picassosr.readthedocs.io/en/latest/others.html).
- Several dialogs in Render, Localize and Simulate are now scrollable (*experimental*).
- SPINNA fix automatic area detection from picked localizations.
- Render add dependency `imageio[ffmpeg]` for building animations.
- Render allow for loading pick regions by dropping a .yaml file onto the window.
- Render improve zooming with mouse wheel (Ctrl/Cmd + wheel).
- Fast rendering automatically adjusts constrast.
- Localize show scale bar function added.
- Localize plotted ROI remains the same when zooming in/out and panning.
- Localize Gauss MLE saves number of iterations and fit log-likelihood.
- DBSCAN accepts min. no. of localizations per cluster.
- Cluster center calculations calculate arithmetic mean, not weighted mean.
- Other bug fixes and minor improvements.

## 0.8.4

- SPINNA - easy fitting of labeling efficiency.
- GUI docstrings added in all scripts; cleaned up docstrings in Picasso modules.
- Render: pick size chosen in nm, not camera pixels.
- Code clean up (flake8 compliant).
- Other bug fixes.

## 0.8.3

- Design: fix export plates and pipetting schemes.
- Design: set default biotin excess to 25 (previously set to 1).
- Render by property allows different colormaps.
- Removed `lmfit` dependency.
- Fix cluster centers bug from v0.8.2.

## 0.8.2

- Added docstrings and data types in all modules (`postprocess`, `simulate`, `render`, `nanotron`, `localize`, `lib`, `io`, `imageprocess`, `gaussmle`, `gausslq`, `design`, `clusterer`, `aim`, `avgroi` and `zfit`).
- Fix one click installer issues for non-administrator users.
- Render allows for saving picked localizations in a separate file for each pick.
- Remaining time estimate in the progress dialog.
- Fix garbage collection when openinging `.nd2` files in Localize.
- Fix 3D rotation window for a polygon pick.
- Render minimap - the zoom-in window is always visible.
- Other small fixes and improvements.

## 0.8.1

- Added `n_events` to cluster centers, i.e., number of binding events per cluster.
- .yaml files contain Picasso version number for easier tracking.
- Improved fiducial picking.
- Bug fixes and other cosmetic changes.

## 0.8.0

- **New module SPINNA for investigating oligormerization of proteins**, [DOI: 10.1038/s41467-025-59500-z](https://doi.org/10.1038/s41467-025-59500-z).
- **NeNA bug fix - old values were (usually) too high by a ~sqrt(2)**.
- NeNA bug fix - less prone to fitting to local maximum leading to incorrect values.
- NeNA plot - displays distances in nm.
- Fiducial picking - filter out picks too few localizations (80% of the total acquisition time).
- `picasso csv2hdf` uses pandas to read .csv files.
- Bug fixes.

## 0.7.5

- Automatic picking of fiducials added in Render: `Tools/Pick fiducials`.
- Undrifting from picked moved from `picasso/gui/render` to `picasso/postprocess`.
- Plugin docs update.
- Filter histogram display fixed for datasets with low variance (bug fix).
- AIM undrifting works now if the first frames of localizations are filtered out (bug fix).
- 2D drift plot in Render inverts y axis to match the rendered localizations.
- 3D animation fixed.
- Other minor bug fixes.

## 0.7.1-4

- SMLM clusterer in picked regions deleted.
- Show legend in Render property displayed rounded tick label values.
- Pick circular area does not save the area for each pick in localization's metadata.
- Picasso: Render - adjust the scale bar's size automatically based on the current FOV's width.
- Picasso: Render - RESI dialog fixed, units in nm.
- Picasso: Render - show drift in nm, not camera pixels.
- Picasso: Render - masking localizations saves the mask area in its metadata.
- Picasso: Render - export current view across channels in grayscale.
- Picasso: Render - title bar displays the file only the names of the currently opened files.
- CMD implementation of AIM undrifting, see `picasso aim -h` in terminal.
- CMD localize saves camera information in the metadata file.
- Other minor bug fixes.

## 0.7.0

- Adaptive Intersection Maximization (AIM, doi: 10.1038/s41592-022-01307-0) implemented.
- Z fitting improved by setting bounds on fitted z values to avoid NaNs.
- CMD `clusterfile` fixed.
- Picasso: Render 3D, rectangular and polygonal pick fixed.
- `picasso.localize.localize` fixed.
- default MLE fitting uses different sx and sy (CMD only).

## 0.6.9-11

- Added the option to draw polygon picks in Picasso: Render.
- Save pick properties in Picasso: Render saves areas of picked regions in nm^2.
- Calibration .yaml file saves number of frames and step size in nm.
- `picasso.lib.merge_locs` function can merge localizations from multiple files.
- Mask dialog in Picasso: Render saves .png mask files.
- Mask dialog in Picasso: Render allows to save .png with the blurred image.
- Picasso: Localize - added the option to save the current view as a .png file.
- Picasso: Render - functions related to picking moved to `picasso.lib` and `picasso.postprocess`.
- Picasso: Render - saving picked localizations saves the area(s) of the picked region(s) in the metadata file (.yaml).
- Documentation on readthedocs works again.

## 0.6.6-8

- GUI modules display the Picasso version number in the title bar.
- Added readthedocs requirements file (only for developers).
- No blur applied when padding in Picasso: Render (increases speed of rendering).
- Camera settings saved in the .yaml file after localization.
- Picasso: Design has the speed optimized extension sequences (Strauss and Jungmann, Nature Methods, 2020).
- Change matplotlib backend for macOS (bug fix with some plots being unavailable).
- .tiff files can be loaded to Localize directly, *although the support may limited!*.
- Bug fix: build animation does not trigger antivirus, which could delete Picasso (one click installer only).
- Bug fix: 2D cluster centers area and convex hull are saved correctly.
- Bug fix: rectangular picks.

## 0.6.3-5

- Dependencies updated.
- Bug fixes due to Python 3.10 and PyQt5 (listed below).
- Fix RCC error for Render GUI (one click installer) (remove tqdm from GUI).
- Fix save pick properties bug in Picasso Render GUI (one click installer).
- Fix render render properties bug in Picasso Render GUI (one click installer).
- Fix animation building in Picasso Render GUI (one click installer).
- Fix test clusterer HDBSCAN bug.
- Fix .nd2 localized files info loading (full loader changed to unsafe loader).
- Fix rare bug with pick similar zero division error.
- Update installation instructions.

## 0.6.2

- Picasso runs on Python 3.10 (jump from Python 3.7-3.8).
- New installation instructions.
- Dependencies updated, meaning that M1 should have no problems with old versions of SciPy, etc.
- Localize: arbitrary number of sensitivity categories.
- Picasso Render legend displays larger font.
- Picasso Render Test Clusterer displays info when no clusters found instead of throwing an error.
- Calling clustering functions from `picasso.clusterer` does not require camera pixel size. Same applies for the corresponding functions in CMD. *Only if 3D localizations are used, the pixel size must be provided.*.
- HDBSCAN is installed by default since it is distributed within the new version of `scikit-learn 1.3.0`.
- Screenshot `.yaml` file contains the list of colors used in the current rendering.
- Render scale bar allows only integer values (i.e., no decimals).
- Localize .ims file fitting bug solve.

## 0.6.1

- **Measuring in the 3D window (Measure and scale bar) fixed (previous versions did not convert the value correctly)**.
- Localize GUI allows for numerical ROI input in the Parameters Dialog.
- Allow loading individual .tif files as in Picasso v0.4.11.
- RESI localizations have the new column `cluster_id`.
- Building animation shows progress (Render 3D).
- Export current view in Render saves metadata; An extra image is saved with a scale bar if the user did not set it.
- (**Not applicable in 0.6.2**) Clustering in command window requires camera pixel size to be input (instead of inserting one after calling the function).
- Bug fixes.

## 0.6.0

- New RESI (Resolution Enhancement by Sequential Imaging) dialog in Picasso Render allowing for a substantial resolution boost, (*Reinhardt, et al., Nature, 2023.* DOI: 10.1038/s41586-023-05925-9).
- **Remove quantum efficiency when converting raw data into photons in Picasso Localize**.
- Input ROI using command-line `picasso localize`, see [here](https://picassosr.readthedocs.io/en/latest/cmd.html).

## 0.5.7

- Updated installation instructions.
- (H)DBSCAN available from cmd (bug fix).
- Render group information is faster (e.g., clustered data).
- Test Clusterer window (Render) has multiple updates, e.g., different projections, cluster centers display.
- Cluster centers contain info about std in x,y and z.
- If localization precision in z-axis is provided, it will be rendered when using `Individual localization precision` and `Individual localization precision (iso)`. **NOTE:** the column must be named `lpz` and have the same units as `lpx` and `lpy`.
- Number of CPU cores used in multiprocessing limited at 60.
- Updated 3D rendering and clustering documentation.
- Bug fixes.

## 0.5.5-6

- Cluster info is saved in `_cluster_centers.hdf5` files which are created when `Save cluster centers` box is ticked.
- Cluster centers contain info about group, mean frame (saved as `frame`), standard deviation frame, area/volume and convex hull.
- `gist_rainbow` is used for rendering properties.
- NeNA can be calculated many times.
- Bug fixes.

## 0.5.0-4

- 3D rendering rotation window.
- Multiple .hdf5 files can be loaded when using File->Open.
- Localizations can be combined when saving.
- Render window restart (Remove all localizations).
- Multiple pyplot colormaps available in Render.
- View->Files in Render substantially changed (many new colors, close button works, etc).
- Changing Render's FOV with W, A, S and D.
- Render's FOV can be numerically changed, saved and loaded in View->Info.
- Pick similar is much faster.
- Remove localization in picks.
- Fast rendering (display a fraction of localizations).
- .txt file with drift can be applied to localizations in Render.
- New clustering algorithm (SMLM clusterer).
- Test clusterer window in Render.
- Option to calculate cluster centers.
- Nearest neighbor analysis in Render.
- Numerical filter in Filter.
- New file format in Localize - .nd2.
- Localize can read NDTiffStack.tif files.
- Docstrings for Render.
- Sensitivity is a float number in Server: Watcher.
- [Plugins](https://picassosr.readthedocs.io/en/latest/plugins.html) can be added to all Picasso modules.
- Many other improvements, bug fixes, etc.

## 0.4.6-11

- Logging for Watcher of Picasso Server.
- Mode for multiple parameter groups for Watcher.
- Fix for installation on Mac systems.
- Various bugfixes.

## 0.4.2-5

- Added more docstrings / documentation for Picasso Server.
- Import and export for handling IMS (Imaris) files.
- Fixed a bug where GPUFit was grayed out, added better installation instructions for GPUfit.
- More documentation.
- Added dockerfile.

## 0.4.1

- Fixed a bug in installation.

## 0.4.0

- Added new module "Picasso Server".
