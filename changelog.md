# Changelog

Last change: 22-OCT-2026 CEST

## 1.0.0

**TODO**: Describe the general overview - fast render, documentation etc. Highlight that there's a lot of backward incompatible changes for `picassosr` users. Accessibility? Standardized API - progress, pick diameter, what else?

### General
- **Nanometers everywhere, and your files keep working.** Camera pixels are no longer a unit in Picasso: coordinates, precisions, spot widths, pick positions and sizes, drift, distances and blur widths are in nm, in memory, in every file and in every dialog, command line option and function argument (`z` was in nm already). Every file written by an earlier version opens in 1.0 as before. See the [documentation](https://picassosr.readthedocs.io/en/latest/files.html#files-legacy) and, for scripts and command lines, the [migration guide](https://picassosr.readthedocs.io/en/latest/migration.html).
- **Stability policy**: from 1.0 on, Picasso follows semantic versioning for its public API (the functions and classes on the API pages), the command line, the file formats and the plugin hooks. Deprecated names keep working, with a warning, for at least two minor releases. Supported Python versions and core dependencies follow [SPEC 0](https://scientific-python.org/specs/spec-0000/): at least three years for each Python version, two years for NumPy, SciPy, etc. See the [documentation](https://picassosr.readthedocs.io/en/latest/versioning.html).
- The command line has one style for its commands and options, words joined by hyphens: `undrift-fiducials`, `nearest-neighbor`, `cluster-filter`, `smlm-cluster`, `group-properties` and `pair-correlation` (were `undrift_fiducials`, `nneighbor`, `clusterfilter`, `smlm_cluster`, `groupprops` and `pc`), and options such as `--r-max`, `--bin-size`, `--keep-index` and `--from-file`. The old names keep working until Picasso 2.0 and print a warning with the new name. See the [documentation](https://picassosr.readthedocs.io/en/latest/cmd.html#cmd-renamed).
- New [documentation](https://picassosr.readthedocs.io/en/latest) that is user-friendly, intuitive and thoroughly explains every Picasso functionality. Quick start page was added with minimal introduction to Picasso. Python API can now be found by a simple browser search in the documentation.
- New appearance of all Picasso windows: a light and a dark theme that follow the operating system, with a selectable accent color, font size and density (`File > Appearance...`); the platform's own style remains available as *Native*. See the [documentation](https://picassosr.readthedocs.io/en/latest/others.html#appearance).
- Toolbars in Render (also its 3D view), Localize, Filter and Average with the most used actions of the menus, see the [documentation](https://picassosr.readthedocs.io/en/latest/others.html#toolbars).
- Icons added to menu actions and buttons.
- The sections of the long dialogs collapse and expand by clicking their header, and each dialog remembers which sections are open across sessions (`Expanded sections` in the [user settings](https://picassosr.readthedocs.io/en/latest/others.html#user-settings-file)).
- A Picasso window left open but unused releases its memory after 30 minutes without input. Set the time with `Release memory after idle (min)` in the [user settings](https://picassosr.readthedocs.io/en/latest/others.html#user-settings-file) where more details are explained.
- Removed support for Python 3.10 (Python 3.11–3.14 are supported).
- All docstrings of public functions have been improved to match the NumPy style.
- Every file Picasso writes records the version of its format (`picasso_format`, now 1.0, as an attribute of .hdf5 files, a key of .yaml files and a header line of drift .txt files) and the version of Picasso that wrote it. Format 1.0 holds every length in nm. Picasso refuses files of a newer major format version and brings files written before 1.0 to the current format (camera pixels become nm, the column `likelihood` is read as `log_likelihood`). See the [documentation](https://picassosr.readthedocs.io/en/latest/files.html#files-format-versions).
- Loading localizations checks the minimum requirements of the format and names what is missing: the columns `frame`, `x`, `y`, `lpx` and `lpy` and the metadata keys `Width`, `Height`, `Frames` and `Pixelsize`. `picasso.io.validate_locs` runs the same check, e.g., before saving files from other software. A file that lacks only `Pixelsize` raises `picasso.io.MissingPixelsizeError`, and the GUIs ask for the pixel size. See the [documentation](https://picassosr.readthedocs.io/en/latest/files.html#files-minimum-requirements).
- Picasso reports how many localizations were discarded on loading (NaN values, outside the field of view, negative values; Render shows a message) and points out a wrong pixel size or canvas when most of them lie beyond the field of view. Previously, such a file opened empty without explanation.
- Picasso warns when the dark times of a file may have been computed before Picasso 0.11.3, when they were one frame longer.
- The .yaml files are read safely: no code named in a file is run. Previously, opening localizations next to a crafted .yaml file could run arbitrary code. This holds for every .yaml file Picasso opens, also picks, masks, 3D calibrations, nanoTRON model metadata and the camera configuration (`config.yaml`). Picasso now writes metadata and calibrations as plain YAML; files with the Python-specific tags that earlier versions wrote (e.g., for numpy numbers) are still read. See the [documentation](https://picassosr.readthedocs.io/en/latest/files.html#files-yaml).
- Each processing step records `Picasso version` and `Step` (e.g., `Filter`) in the metadata, made by `picasso.lib.step_info`; `Generated by` reads `Picasso v<version> <step>` throughout. Filter finds its steps in the metadata by `Step`.
- **One pattern for the Python functions on localizations:** they take the localizations and their metadata first, `f(locs, info, ...)`, and every other argument by name, and those that return new localizations also return their complete metadata, ready to be saved: `info` with an entry recording the step and its parameters appended, e.g., `linked, linked_info = postprocess.link(locs, info, r_max=10)`. The command line writes the same entries, as it uses these functions. `picasso.lib.append_metadata` records steps of your own in the same way. See the [documentation](https://picassosr.readthedocs.io/en/latest/api/index.html#api-locs-info).
- Test version documentation online.
- The plugin store asks before uninstalling a plugin, and says so when a plugin no longer in the online registry cannot be installed again.
- Fixed: without a connection to the online plugin registry, the plugin store listed every installed plugin as "Installed (unlisted)", no longer in the registry.

### Render
- **Render has been largely rewritten for fast and interactive rendering of localizations even for large, multiplex datasets. GPU via the package ``wgpu`` has been added for cross-platform support**. More details in the section **Technical details on Render update** below. See these links for the relevant user guides: [navigation](https://picassosr.readthedocs.io/en/latest/render.html#render-navigation), [user_settings](https://picassosr.readthedocs.io/en/latest/render/performance.html#render-cpu-usage), [GPU](https://picassosr.readthedocs.io/en/latest/render/performance.html#render-gpu-rendering).
- `View > 3D view` (Ctrl+Shift+R, replacing *Update rotation window*) opens the rotation window on the single selected pick or, without a pick, on the current field of view, so any region can be inspected in 3D by zooming to it; pressed again on unchanged content it only raises the window.
- Linked windows that share user-selected attributes, see [documentation](https://picassosr.readthedocs.io/en/latest/render/menu-view.html#render-new-linked-window).
- New blur methods *Adaptive Histogram (Quad-Tree)* and *Jittered Triangulation* (Baddeley, Cannell & Soeller, *Microsc. Microanal.* 2010). See the [blur documentation](https://picassosr.readthedocs.io/en/latest/render/display-settings.html#render-blur).
- The display settings (main and 3D window) show the minimum blur only for the Gaussian blur methods that use it.
- New action to move xy positions of localizations with a mouse. See the [documentation](https://picassosr.readthedocs.io/en/latest/render/menu-tools.html#render-move).
- Apply expression to localizations expands the canvas (metadata's `Height` and `Width`) if x and y positions are out of range.
- Overlay of image files (`.png` and `.tif`), see the [documentation](https://picassosr.readthedocs.io/en/latest/render/menu-view.html#render-overlay-image).
- Editable appearance of the tools (e.g., picks), see the [documentation](https://picassosr.readthedocs.io/en/latest/render/menu-tools.html#render-tools-settings).
- Per-channel contrast for multichannel data in `View > Files...`, replacing the relative intensity. See the [documentation](https://picassosr.readthedocs.io/en/latest/render/menu-view.html#render-channel-contrast).
- Smooth transitions in 3D animations: the motion eases in and out and passes through the positions without sharp turns, see the [documentation](https://picassosr.readthedocs.io/en/latest/render/3d.html#render-animation).
- Faster circular picking of localizations.
- Undrifting from picked localizations is fast and memory-efficient for many picks.
- Faster FRC.
- FRC in several random ROIs (*Info* dialog, *FRC in several ROIs*).
- `picasso.render` is distributed as a package (`kernels`, `geometry`, `splat`, `scene`, `overlays_qt`, `animation`, `backend`, `gpu`); every former `picasso.render.*` name is still importable from `picasso.render`.
- 2D/3D grid available for reference, either as the full grid or as axes with ticks and labels only.
- The axes icon in the bottom left corner of the 3D view was removed (along with `View > Show/hide rotation`).
- Rectangular pick minimum length set to 5 display pixels.
- The zoom-in rectangle (main and 3D window) can be dragged in any direction and canceled with a right click or Esc. See the [documentation](https://picassosr.readthedocs.io/en/latest/render.html#render-navigation).
- The View menu no longer lists Left/Right/Up/Down and Zoom in/out (their keyboard shortcuts still work); *Filter picks by number of localizations* is now *Filter picks by count*.
- Long operations run in the background and can be canceled: undrifting (AIM, RCC, from picked), DBSCAN, HDBSCAN, SMLM clusterer, G5M, RESI, NeNA, FRC in ROIs, pick statistics and saving pick properties. The windows keep repainting while they run.
- Link localizations can be run on all channels sequentially.
- Coordinate-based colocalization (CBC, Malkusch, et al., *Histochem Cell Biol* 2012). See the [documentation](https://picassosr.readthedocs.io/en/latest/render/analysis.html#render-cbc).
- New `Postprocess > Spatial statistics` submenu for analyzing how the molecules of one channel, or of several channels with respect to each other, are distributed, including Ripley's H function (Ripley, *J R Stat Soc B* 1977), the radial distribution function (RDF) and the nearest neighbor distance distributions of all channels, each compared with random simulations. See the [documentation](https://picassosr.readthedocs.io/en/latest/render/menu-postprocess.html#render-menu-spatial-statistics).
- Stitching of tiles recorded at different stage positions into one dataset. See the [documentation](https://picassosr.readthedocs.io/en/latest/render/menu-postprocess.html#render-stitch-tiles).
- Basic single-particle tracking (SPT), following Martens, et al., *Front. Bioinform.* 2021: nearest-neighbor tracking of localizations (`Postprocess > Single-particle tracking`), tracks drawn as lines (`Display settings > Tracks`) and diffusion analysis by the mean squared displacement and the jump distances, with a summary of every track. See the [documentation](https://picassosr.readthedocs.io/en/latest/render/tracking.html). *Note: this is an experimental feature, feedback is welcome*.
- Clear measurements added (similar to Clear picks)
- Files dialogs widgets are grayed out if they do not apply at the given moment.
- Faster G5M.
- Files dialog is now shown as a table with indices for easier navigation. Channels can be reordered and the default ordering of colors can be adjusted. `Load colors` works with any number of channels.
- `View > Color vision preview` shows the main and 3D windows as people with deuteranopia, protanopia or tritanopia see them; saved and exported images keep their colors. The simulation is also available in Python as `render.simulate_color_vision`. See the [documentation](https://picassosr.readthedocs.io/en/latest/render/menu-view.html#render-color-vision-preview).
- `Select picks (XYZ scatter)` and `(XYZ scatter, 4 panels)` color z with the perceptually uniform, colorblind-friendly *plasma* colormap instead of *jet*.
- `Colorblind-safe colors` checkbox in the Files dialog colors the channels magenta, green, cyan, blue, and so on, which people with red-green color blindness can tell apart. See the [documentation](https://picassosr.readthedocs.io/en/latest/render/menu-view.html#render-default-colors).
- The Info dialog shows the median localization precision of any loaded channel.
- *Align channels* can remove the localizations that lie outside the overlap of the aligned channels. See the [documentation](https://picassosr.readthedocs.io/en/latest/render/menu-postprocess.html#render-align-non-overlapping).
- Optional confirmation before `File > Remove all localizations`, turned on with the user setting `Confirm removing all localizations` (off by default). See the [documentation](https://picassosr.readthedocs.io/en/latest/render/menu-file.html#render-remove-all-localizations).
- Fixed: closing a channel in Render did not remove its drift data, so the drift of the channels after it was misaligned.
- Fixed: 3D histogram rendering scaled z unnecessarily.
- Fixed NeNA (and FRC) overwriting group columns of localizations.
- Fixed: DBSCAN, HDBSCAN and SMLM clusterer saved the cluster areas as `*_centers_areas.csv` instead of `*_areas.csv` when the cluster centers were saved, too.
- Fixed: *Export FOV as .ims* ignored the image extents of the loaded metadata and used only the last channel's metadata.
- Fixed: the `locs` column of saved pick properties held the number of binding events (the same as `n_events`) instead of the number of localizations in the pick (`picasso.postprocess.pick_kinetics`, `pick_properties`).
- Fixed: scale bar in 3D animations.
- Fixed: *Align channels* (RCC) failed for channels with different image sizes.
- Fixed: Render's `Export ROI for Imaris` ignored the position of the .ims movie that localizations were loaded from with `File > Open movie` (single channel), placing the export at the origin instead of over the movie.
- Fixed: *Undrift by RCC* suggested a segmentation of 1000 frames for movies of 1000 to 3500 frames, which gives fewer than the 4 segments RCC needs, so it failed. It now suggests at most a quarter of the frames, and a segmentation that leaves fewer than 4 segments (also in `picasso undrift` and `picasso.postprocess.undrift`) is explained instead of failing inside RCC.
- Fixed: *Test clusterer* failed when no cluster was found, instead of saying so, and when the display options or projections were changed before the first test. `picasso.clusterer.find_cluster_centers` failed in the same way for localizations without clusters, e.g., when DBSCAN, HDBSCAN or the SMLM clusterer found none and the cluster centers were to be saved.
- Fixed: *Analyze cluster* (k-means in picks) started clustering, and failed, when the number of clusters was canceled, and saved the pick properties (`_properties.hdf5`) in the current folder when saving was canceled. Also, the new group ids of the clusters are integers instead of floats.
- Fixed: showing the FRC plot failed when the FRC calculation it starts was canceled.
- Fixed: *Export channels in grayscale* with a suffix of another format raised an error instead of naming the possible formats; *Export view manually* left the contrast changed, and its fields unresponsive, when the DPI of a PDF was canceled; the export of z slices of separate channels showed all channels afterwards.
- Fixed: *Select picks* raised an error when a pick was rejected again after going back to it, *Select picks (trace)* left the trace window open when canceled, and with all channels shown a pick of exactly 3 or 4 localizations was not drawn in the channel colors.
- Fixed: closing a pick of *Select picks (XY / 3D / XYZ projections)* or *Analyze cluster* with Escape or the close button removed the pick; it now stops the selection, as *Cancel*.
- Fixed: in the custom colormap editor, a new or duplicated colormap named like the selected one overwrote it, and *Rename* suggested another name; an unusable custom colormap (.npy) raised an error instead of a warning.
- Fixed: canceling G5M on 3D data before loading a 3D calibration raised an error; *Mask image* raised an error when the channel choice was canceled and when saving the masked localizations of any channel but the first.
- Fixed: *Nearest neighbor analysis* failed for one neighbor in another channel and between a 3D and a 2D channel (now compared in 2D); a too high number in *Move to pick* raised an error; canceling *Unfold groups (square)* after the picks were turned into groups removed all localizations outside the picks.
- Fixed: resetting a channel title in the Files dialog showed "reset" for short file paths; *Export current view* with all channels hidden and *Save picked localizations* (also separately) without picks raised an error; *Filter picks by count* kept every histogram figure in memory.
- Fixed: *Export ROI for Imaris* showed pixels brighter than the contrast maximum as dark, and failed, after deleting an existing file, with all channels hidden or with channels with and without z.

### Localize
- New spot identification method: B-spline wavelet segmentation (Izeddin et al., *Opt. Express* 2012), see the [documentation](https://picassosr.readthedocs.io/en/latest/localize/identification.html#localize-wavelet).
- New column `reduced_chi_square` for MLE and least-squares fits: the goodness of fit normalized for the box size and the photon counts, about 1 for a good fit (`picasso.localize.reduced_chi_square`).
- The quantum efficiency is no longer shown in the Parameters dialog. A `Quantum Efficiency` entry in the camera config, `--qe` in `picasso localize` and a `Qe` key in `camera_info` are still accepted and ignored. `camera_parameters` of the movie classes in `picasso.io` no longer returns `qe`. The `Emission Wavelength` dropdown now lists the wavelengths of `Channel Device` > `Emission Wavelengths` and of the wavelength-keyed calibrations in the config (it appeared only when the config had a `Quantum Efficiency` mapping before, so wavelength-keyed calibrations were not selected with the template config).
- Imaris `.ims` movies can now be opened on macOS and Linux, not only Windows. Exporting `.ims` files (Render's `Export ROI for Imaris`) remains Windows only. `hdf5plugin` is now installed on all platforms.
- Every calibration records how its spots were identified, under `Identification`.
- Fixed: ratiometric multichannel spline fitting ignored the per-pixel sCMOS variance of the camera calibration.
- Fixed: with an sCMOS camera calibration and a CUDA GPU, spline fits of movies with more than about 1.5-2 million localizations reported wrong localization precisions (`lpx`, `lpy`, `lpz`, ...) beyond the first ~2 million, which were computed with the readout variance of other localizations.
- Fixed: `Height` and `Width` were swapped in the metadata of non-square `.ims` movies.
- Fixed drift estimation when adding to database + database building.
- Fixed: canceling the channel selection when opening a multi-channel IMARIS .ims movie raised an error instead of loading the first channel, as for .czi and .lif.
- Fixed: indexing a TIFF movie of a single file with a tuple (e.g., `movie[2, 1]` or `movie[1:3, :, 5]`) raised a `TypeError`, as did slicing the frames of a MetaMorph .stk file with a tuple. `movie[..., x]` now indexes the last axis, as in NumPy, for all movie formats.
- Fixed: the `shape` of non-square Nikon .nd2 movies swapped the height and width.
- Fixed: writing TIFF and .stk movies to raw files (`tofile`) swapped the bytes of big-endian movies, and of all movies when no byte order was given.
- Fixed: starting a channel registration or a spline PSF calibration while one was still running closed Localize. Localize now says that one is already running.
- Fixed: a 3D (astigmatism) calibration with more frames per step than the movie has, or a z binning that leaves fewer than 7 axial bins, raised an error instead of a warning saying what to change.
- Fixed: identifications loaded from picks, localizations or saved identifications lost the spots right of x = number of frames (the check against the field of view took the frame count for the width); opening a file that is not HDF5 as identifications raised an error instead of saying so.
- Fixed: entering a frame range for a `.raw` movie raised an error; the message that no spots were found on the channel sum disappeared at once.
- Localize tells why a fit failed (the error, in a message box) instead of only "Aborted.", and an identification that finds no spots says so, instead of leaving its progress in the status bar, and drops the spots of an earlier identification.
- Fixed: Localize did not open, or crashed when the camera was selected, if a camera of the camera config has no `Sensitivity` or `CameraPriority` names a camera missing from `Cameras`.
- Fixed: `picasso localize` stopped at the first movie that could not be undrifted (e.g., too few frames), skipping the other movies; it now says so and goes on, keeping the localizations.
- Fixed: `camera_parameters` of multi-file TIFF movies (`TiffMultiMap`, also Micro-Manager separate files and concatenated TIFF movies) failed unless the movie's `info()` had been read first, and of .nd2 movies whose metadata could not be read raised a `TypeError` instead of the `KeyError` for an unknown camera.

### Filter
- Filtering by expression: `Filter > Filter by expression...` for universal filtering with comparisons, logic, arithmetic, functions and statistics over the shown localizations. See the [documentation](https://picassosr.readthedocs.io/en/latest/filter.html#filter-expression).
- The metadata of a filtered file lists its filter steps in the order they were applied.
- Filters from metadata can also be applied by dragging the .yaml file into the Filter window.
- Filter shows 1D histograms of both variablies in the 2D histogram filtering.

### **Backward incompatible changes:**
The [migration guide](https://picassosr.readthedocs.io/en/latest/migration.html) lists what changes for you, with the replacement of each setting, command and function.
- **Lengths in nm** (see *General* above): in the files Picasso writes, which earlier versions would read as camera pixels, and in every dialog, command line option, function argument and return value. Defaults that were in camera pixels are fixed nm values, e.g., `link` 10 nm, `aim` 20 and 60 nm, `pair-correlation` 10 and 1000 nm. `Pixelsize` is a required metadata key. The `pixelsize` arguments that only converted units are removed (e.g., of `picasso.clusterer.dbscan` and the commands `dbscan`, `hdbscan` and `smlm-cluster`), while `picasso.localize.localize`, `fit`, the `locs_from_fits_*` functions and the lateral calibrations require the pixel size. Render's *Zoom* is replaced by *Display width (µm)*, and its *Camera pixel size* by the `Pixelsize` of each channel. See the [guide](https://picassosr.readthedocs.io/en/latest/migration.html#migration-python-units).
- **Circular picks are sized by their diameter** in every function: `picasso.postprocess.picked_locs` and `undrift_from_fiducials` took the radius, and `picasso.nanotron` takes `pick_diameter`. `picasso undrift-fiducials` records `Pick diameter (nm)`.
- **Identification settings** are one `picasso.localize.IdentificationParameters` (the method, `NetGradient` or `WaveletParameters`, the box and the filters) in `picasso.localize.identify`, `localize` and `identify_multichannel_sum`, and the calibrations take an `identification_method` in place of `minimum_ng`. `localize` takes only the movie positionally.
- **Progress** is reported through one `progress` argument in every function: `"console"`, None or a tracker such as `picasso.lib.ProgressDialog`. It replaces `progress_callback`, `callback`, `callback_parent` and the separate callbacks per step.
- **Removed:** Picasso: Average3 (`picasso average3`); the modules `picasso.gausslq` and `picasso.gaussmle` (use `picasso.localize` and `picasso.fitting`); `picasso.localize.fit2D`, `localize_3D`, `locs_from_fits` and `locs_from_fits_gauss_gpu`; `picasso.lib.append_to_rec` and `remove_from_rec`; the `return_info` arguments; `picasso.postprocess.cluster_combine` and `cluster_combine_dist` with their commands; `picasso.io.load_clusters` and the cluster tables written before Picasso 0.5; `picasso.render.scale_intensities`; *Tools > Fast rendering* in Render.
- **Renamed:** the functions of `picasso.simulate` and `picasso.design` have snake_case names (e.g., `convertMovie` → `convert_movie`), the numba functions of `picasso.lib` lost their `_numba` suffix (e.g., `locs_at_numba` → `locs_in_circle`), `picasso.spinna.load_structures` is `picasso.io.load_structures`, and the sound notification functions of `picasso.lib` are in `picasso.notifications`.
- **Functions on localizations** take only `locs` and `info` positionally and return their metadata, too (see *General* above): `postprocess.link`, `undrift` (`locs, info, drift`; was `drift, locs`), `align` and the other steps return `locs, info` first, `clusterer.cluster`, `dbscan` and `hdbscan` the whole metadata instead of only their entry, `g5m.g5m` returns `centers, info, clustered_locs`, and `lib.apply_filter_steps` takes the filter steps as `recipe`. The metadata the commands `link`, `undrift`, `undrift-fiducials` and `align` write changed accordingly. See the [guide](https://picassosr.readthedocs.io/en/latest/migration.html#migration-python-locs-info).
- **Alignment:** `picasso.postprocess.align` repeats the RCC alignment until it converges, also aligns from picks (`method="picked"`) and no longer shifts its input in place (`apply_shifts` is removed), and neither does `align_from_picked`. The returned shifts are ordered x first; `align_from_picked` returned them y first.
- **Per-channel contrast** replaces the relative intensity of channels in Render and the `relative_intensities` of `picasso.render.render_scene` and `build_animation` (`channel_contrast`). `build_animation` takes positions only as `(rotation, viewport)`.
- **Loading files:** the loaders of `picasso.io` take no `qt_parent`, show no dialogs and print nothing; `load_locs` raises `picasso.io.LocsFormatError` (a `ValueError`) instead of a `KeyError`; metadata read from .yaml files holds plain Python values instead of NumPy types.
- Deprecation warnings are `FutureWarning`s instead of `DeprecationWarning`s, so that scripts show them.
- Picasso Server only accepts connections from the computer it runs on; `picasso server --network` allows other computers. Previously, anyone who could reach the computer could use it, including the watcher's custom command. It is an optional extra of the Python package: `pip install "picassosr[server]"`.
- For plugins: Render's `View.pixelsize` is replaced by the read-only `View.pixelsizes` (one per channel), `DatasetDialog.intensitysettings` by `contrast_low` and `contrast_high`, and `View.read_relative_intensities` by `View.read_channel_contrast`.

### Others
- Many plots are now customizable.
- High contrast (`File > Appearance... > Contrast`): black and white text and backgrounds, and frames in the color of the text around buttons, input fields and tables. By default it follows macOS's *Increase contrast* and Windows' contrast themes. See the [documentation](https://picassosr.readthedocs.io/en/latest/others.html#appearance).
- Screen readers (VoiceOver, Narrator, Orca) announce the input fields of all Picasso windows by their labels, fields in tables by their column and icon-only buttons by their tooltip. See the [documentation](https://picassosr.readthedocs.io/en/latest/others.html#accessibility). Note this is an experimental feature, if there is room for improvement, please raise a [GitHub issue](https://github.com/jungmannlab/picasso/issues)
- Buttons that show a state (e.g., a loaded file in SPINNA) show an icon besides their color, so that the state does not rely on color alone; screen readers read it out.
- Code readability clean ups (flake8).
- For developers: `picasso.deprecation` declares deprecations with the version that may remove them, and `tests/test_stability.py` compares the public names and the command line with snapshots and fails once a deprecated name is due for removal. See the [documentation](https://picassosr.readthedocs.io/en/latest/versioning.html#versioning-developers).
- The spatial index stored in localization .hdf5 files (`/render_index`). See the [documentation](https://picassosr.readthedocs.io/en/latest/files.html#spatial-index).
- The user settings file (`~/.picasso/settings.yaml`) is no longer lost when it cannot be read. Also, the [documentation](https://picassosr.readthedocs.io/en/latest/others.html#user-settings-file) has been added.
- The Windows one-click installer's *GPU* edition is renamed *CUDA*, similarly the installed folders were renamed etc.
- Fixed: combining localizations whose columns differ (e.g. with and without `net_gradient`, spherical and elliptical, least-squares and MLE fits, or 2D and 3D data) silently deleted all localizations of the files lacking a column when saving. `picasso join`, RESI, and the multi-channel saves of Render (picked localizations, also per pick) and its 3D window now drop the columns not all of them have, with a warning.
- Fixed Nanotron's PyQt6 information dialogs.
- Fixed: Design's plate export with colors gave an empty well the color of the well before it, and failed if the first well of a plate was empty.
- Picasso's modules report through Python's `logging` (one logger per module, `picasso.io`, `picasso.localize`, ...) instead of printing, so scripts can show, silence or redirect the messages. Warnings appear as before; the command line also shows the information messages. Output asked for with `verbose=True` is still printed. See the [documentation](https://picassosr.readthedocs.io/en/latest/api/index.html).
- Picasso says when it leaves out metadata it cannot read (e.g., Micro-Manager, CZI or LIF metadata) and when the camera of a movie is missing from the camera configuration, so that gain 1 is used. Before, these happened without a trace.
- Opening a file whose path is too long gives a warning, see [FAQ](https://picassosr.readthedocs.io/en/latest/faq.html#picasso-says-that-the-path-is-too-long).
- Fixed: `picasso.postprocess.remove_locs_in_picks` and `compute_dark_times` modified the localizations passed to them, and `picasso undrift --from-file` ignored the z drift.
- Fixed: `picasso.postprocess.evaluate_picks` changed the warning filters of the whole program, e.g., showing again the `RuntimeWarning`s a script had silenced or turned into errors.
- Fixed: `picasso render` on the command line failed without a settings file, e.g., on a new installation, unless a colormap was given with `-c`.
- Fixed: `picasso smlm-cluster` required `basic_fa` and `radius_z`, although they are optional, and switched the frame analysis on for any `basic_fa`, including `False`. It now takes `yes`/`no` (or `true`/`false`, `1`/`0`).
- Fixed: `picasso undrift-fiducials` failed for movies shorter than 7000 frames: the RCC before picking the fiducials used segments of 2000 frames. It now uses at most a quarter of the frames.
- Fixed: `picasso hdf2csv` wrote the row numbers as an extra unnamed first column.
- Fixed: SPINNA's batch analysis (`picasso spinna -p`) read the `rotation_mode` *None* as missing and used *2D*. Also, the summary's column *File location of experimental data* was misspelled.
- Fixed: the GUIs could crash when quitting (seen on macOS): their windows were left to Python's shutdown, which deletes them in no fixed order.
- Fixed: a camera configuration (`~/.picasso/config.yaml`) that is not valid stopped every Picasso module and command from starting. Picasso now starts without cameras configured and warns; Localize shows why when it opens.
- Fixed: `picasso.nanotron.roi_to_img` hid every error of rendering behind an `UnboundLocalError`, and returned an empty image for a pick without localizations, which failed later. It now raises the error of rendering, or a `ValueError` that names the empty pick.
- Fixed: the command line asked about an available update from a background thread while the command was still running. In batch jobs, e.g., on a computing cluster, this printed an error or crashed Python on exit, and in a terminal the update question could take the answer meant for the command's own question. Picasso now tells about an update after the command has finished and asks only in a terminal; elsewhere it prints how to switch the checks off.
- Fixed: update notifications did not appear until an update check had been recorded, which only a notification did. Also, every command asked GitHub for the latest release even with update checks switched off or snoozed, and the GUIs ignored *Don't check for updates*, *Skip this version* and *Remind me in 7 days*.
- Fixed Picasso Server's watcher, which stopped at its first check of the folder since 0.11.0. Also, its custom command ran only on Windows and only with `$FILENAME` in it; it now runs on macOS and Linux, too, passes the file name as one argument also when it contains spaces, and the logfile shows the exit code of a failed command. The new placeholder `$LOCS` passes the localization files, running the command once per file, e.g., `picasso undrift $LOCS`; the documented examples with `$FILENAME`, the movie, could not work. See the [documentation](https://picassosr.readthedocs.io/en/latest/server.html#server-custom-commands).
- Fixed in Picasso Server: *History*'s filename filter showed no files when it only excluded tags (e.g., `|test`) and all files after a trailing `|`, and its maximum acquisition date left out the files acquired on that day. *Compare* plotted the histograms at shifted positions, left the frames without localizations out of the localizations per frame, and, like *Preview*, offered the localization files of other movies whose names start the same (e.g., `cell10_locs.hdf5` for `cell1.raw`, or every file of the day for names like `2026.01.05_cell.raw`). *Status* failed when a folder added to the database held a movie without localizations.
- Fixed in Picasso Server: the watcher localized no movie (each failed with an error in its log); it also localized movies again that had localization files, localized the same movie at every check when the folder was entered as a relative path, and stopped when Localize exited on a movie. The *Watcher* page failed with 3D in more than one parameter group and when the logfile of a listed watcher was missing.
- Picasso Server's watcher takes the camera pixel size from the movie's metadata, else the camera config, as `picasso localize`, unless one is entered; it used 130 nm for every movie.
- Picasso: Simulate has a new cleaner layout.
- Fixed: Simulate simulated a different photon rate than it showed. Until the power density or the photon detection rate was changed, a new window simulated 5,300 photons per frame (an outdated default) while showing 22,750, and photon rates above 1,000 photons per ms were capped. **Simulations with the default settings are therefore about 4x brighter than before**; set the `Photon detection rate` to ~300 for the previous brightness.
- Fixed: Simulate's `File > Load settings from previous simulation...` failed with an error, leaving the settings half loaded, when the selected .yaml file was not written by Simulate. Such files are now reported and change nothing.
- Fixed: Simulate's `File > Import handles...` crashed for .yaml files (the structure landed in the corner of the field of view, outside the frame) and ignored their z coordinates.
- Fixed in Average: saving more than once shifted the localizations again each time, and the saved file lost localizations; opening a file while averaging broke the averaging (it is now refused); picks of a single localization each crashed on opening.
- Fixed in Simulate: the *Circle* structure sometimes got one docking strand more than the number of labels; the preview could crash when the first structure had no docking strand incorporated; with *Export kinetic data*, multi-round simulations did not save the kinetics of the last round, and the status said the movie was saved after every round.
- SPINNA warns when loaded structures share a title (only the first one of a title can be selected or deleted); the mask threshold stays below the highest probability, so a thresholded mask is never empty; *Fit LE* with *Save fit scores* does not run without a folder, as *Compare models*; a single simulation asks where to save the molecules only once it is known to run.
- Design warns before saving a pipetting scheme with sequences missing from the plate list, and the saved message says how many are missing.
- nanoTRON models are trained on at least two classes, and predicting with a model whose .yaml does not record the pick diameter or the display pixel size asks for them.
- Fixed in SPINNA: deleted molecular targets stayed visible in the Structures tab; opening a structures file or mask that SPINNA did not save crashed instead of saying so; canceling *Load* of experimental data cleared the NND histograms of the plot on screen, so *Save plots* and *Save values* did nothing.
- Fixed in SPINNA: canceling the number of simulations when loading a search space still replaced the search space, and an invalid search space file crashed; invalid NND plot settings (minimum distance above the maximum, bin size larger than the range) crashed after the warning; *Save values* saved nothing for the experimental NNDs of structures without neighbors (e.g., monomers only).
- Fixed in SPINNA: saving the z slices of a 3D mask changed the displayed mask; fitting a 2D simulation to 3D data discarded the z coordinates of the loaded data, so a later 3D simulation was refused, and single 2D simulations of 3D data were scored with 3D distances. `picasso.spinna.MaskGenerator.save_mask(save_png=True)` failed for 2D masks.
- Fixed in Design: the extension table and the folding scheme crashed when every binding site was colored; an erased extension name or sequence was still used for the plates, and an extension with a name but no sequence was accepted; loading a file that is not a design, or non-numeric input in the folding table, raised an error instead of a message.
- Fixed in Design: canceling the pipetting dialog still created the scheme, and choosing a folder there changed the working directory and kept the sequences of a previously chosen folder.
- Fixed in nanoTRON: a model file or its .yaml that could not be read crashed and could leave the new model with the old classes; predicting hung when it failed (e.g., for a model trained with another image size), and failed for picks whose group numbers do not start at 0 or have gaps (e.g., after filtering picks).
- Fixed in nanoTRON: unchecked classes were exported anyway; *Export* and `File > Save` crashed before predicting, and exported earlier predictions under the classes of a model loaded afterwards; the status bar counted the loaded picks one short, and localizations without picks crashed. The pick files of the export are named `<file>_<class>_picks.yaml` (was `<file>__<class>_picks.yaml`), and the export metadata no longer records a probability filter or regrouping that was switched off.
- Fixed in nanoTRON's training dialog: canceling *Save model* wrote `.sav` and `.yaml` files into the working folder; canceling the file dialog of training data, training data without a pick diameter, the learning curve of the lbfgs solver, and a failed training raised errors (a failed training also blocked further training); *Expand training set* showed wrong progress.
- Text search in the user settings dialog (`File > Picasso settings`).
- Contrast slider at the bottom of the Render window, like the one in Localize. Whether Render and Localize show it is set by `contrast_slider` in the user settings (off in Render, on in Localize by default). See the [documentation](https://picassosr.readthedocs.io/en/latest/render/display-settings.html#render-contrast-slider).
- Improved navigation in the SPINNA mask tab.
- Improved status dialog.
- Sound notifications in Average and Simulate (`File > Sound notifications`).
- Added import and export of [PyME](https://python-microscopy.org) (python-microscopy, PYMEVisualize) localizations: raw fit results (`.h5r`) and saved localizations (`.hdf`) are imported, `.hdf` files are exported. See the [documentation](https://picassosr.readthedocs.io/en/latest/render/menu-file.html#render-export-pyme).
- `sqlalchemy` is no longer a dependency: the database of localized files (`~/.picasso/app_0410.db`), which Localize writes to and Picasso Server reads, uses Python's built-in `sqlite3`. Existing databases keep working.
- Fixed: HDBSCAN with a cluster selection epsilon above 0 could fail with `TypeError: only 0-dimensional arrays can be converted to Python scalars`, a bug of scikit-learn with NumPy 2.4 and later ([scikit-learn #33219](https://github.com/scikit-learn/scikit-learn/issues/33219)). The one-click installers ship NumPy 2.3, which avoids it. With NumPy 2.4 and later, Picasso explains the error and how to avoid it. scikit-learn 1.8 and later can be installed again.
- `watchdog` is no longer a dependency: Picasso Server starts Streamlit without its file watcher, which only reloads the app when its source files are edited.
- Menu entries on Linux: `release/create_linux_shortcuts.py` is described in the [installation guide](https://picassosr.readthedocs.io/en/latest/getting-started/installation.html). It now also adds Nanotron and Server, can be run again (it failed when the files existed), creates missing folders, puts the `picasso` command into `~/.local/bin` (`/usr/local/bin` for all users) and deletes everything with `--remove`.
- The MATLAB script in `convert` is replaced by a shorter and corrected one in the [documentation](https://picassosr.readthedocs.io/en/latest/files.html#files-matlab-export).
- CI builds the documentation, failing on warnings, as does Read the Docs, and reports the test coverage of each module; the readme shows the coverage of `master` as a badge. Numba-compiled functions and the CUDA modules, which coverage cannot trace or CI cannot run, are left out of it.
- `CITATION.cff` describes how to cite Picasso: GitHub shows it as *Cite this repository*.
- `dask`, `statsmodels` and `requests`, which Picasso imports, are declared dependencies. Before, they were only installed because other dependencies (`nd2`, `plotly-express`, `streamlit`) need them.
- The one-click installers ship the licenses of all bundled third-party software (Python, Qt, NumPy, ...) in `THIRD-PARTY-LICENSES.txt`. It also lists PyInstaller, whose bootloader and run-time hooks are part of the app, and states that, because of PyQt6 (GPL-3.0), the installation as a whole is distributed under the GPL-3.0, with where the source code of each part is.
- `LICENSES` holds the licenses of all code adapted from other projects: besides Gpufit, Gpuspline and Lucide, also scikit-image (thresholds in `picasso.masking`), SciPy (the bounded minimizer of GPU z fitting), scikit-learn (G5M) and AIM.
- Fixed: negative frame indices (e.g. `movie[-1]`) raised an `IndexError` for movies split across several files.
- Fixed the docstring of `picasso.imageprocess.rcc` ([#716](https://github.com/jungmannlab/picasso/issues/716)).
- Fixed: with `Save metadata in .yaml` off, saving over a file that had a .yaml file kept the outdated .yaml file, which then took precedence over the new metadata on loading. An existing .yaml file is now always overwritten.
- Fixed: NeNA left out the last localizations (up to 99, and all of them for fewer than 100 localizations) and missed a next-frame neighbor at the end of the localizations, while the last localization added a distance to a random bin, and a distance of exactly 200 nm was written outside the histogram. NeNA values change very slightly.
- Fixed: `picasso.postprocess.groupprops`, `pick_properties` and `resi` failed with `progress="console"`; `groupprops` also failed for localizations without the `dark` column, and `pick_properties` for an empty list of picks.
- Fixed: `picasso.postprocess.undrift`, `picasso.imageprocess.rcc`, `picasso.clusterer.cluster_areas`, `picasso.localize.check_drift` and `get_file_summary` printed progress bars when asked for no progress, and `picasso.localize.check_nena` returned NaN when asked to show its progress in the console.
- Fixed: progress bars that stopped one step short of the end, counted items that were still being processed, or started again from zero within one step (e.g., in the identification, the fits, z fitting, AIM and SPINNA).
- Fixed: Average spun at full CPU load in its main process while waiting for the alignment of each iteration.

### Technical details on Render update
- **Multi-threaded CPU rendering**: the render kernels release the GIL and channels are rendered in parallel by a thread pool; a single large channel is split into row chunks rendered in parallel and summed in a fixed order.
- New user setting `cpu_utilization` (`Render` section of `~/.picasso/settings.yaml`, default 0.5): the fraction of the CPU cores rendering may use, for shared workstations. See the [documentation](https://picassosr.readthedocs.io/en/latest/render/performance.html#render-cpu-usage).
- The display pipeline after splatting (contrast, colormaps, channel compositing, 8-bit conversion) runs in fused numba kernels instead of a chain of numpy operations (another speed improvement).
- **GPU rendering in Render**: localizations are rendered on the graphics card (Metal, Direct3D 12 or Vulkan via `wgpu`, any vendor, no CUDA needed) — uploaded once, then every view is computed on the GPU, several times faster than the CPU threads. See the [documentation](https://picassosr.readthedocs.io/en/latest/render/performance.html#render-gpu-rendering).
- `View > Show info` shows the active renderer ("GPU (Apple M4 via Metal)" or "CPU (5 workers)") with a help button that opens the GPU documentation, which lists the requirements and what to check when it says CPU.
- New user setting `max_blur_width` (`Render` section, default 100 nm): localizations with a precision worse than this are not rendered by the individual-precision blur methods (they would only add a faint wide haze while costing most of the render time). `0` or `off` disables the limit.
- **Rendering runs on a background thread** in the main window: panning and zooming never block the interface, a burst of mouse events renders only the newest view, and the last image is shifted or scaled on screen immediately while the new one renders, so dragging feels continuous. Renders cover a 15% margin around the window so small pans need no new render at all; exports still render the exact view.
- While panning and zooming, large datasets are previewed with a subset of the localizations (new user setting `interaction_subsample`, `Render` section: `auto` = at least 500,000 or 10% of the localizations in view, or a fixed number) with the contrast compensated, and sharpened as soon as the mouse pauses.
- Loading localization files no longer needs the *indexing localizations...* step before circular picking, and the spatial index used for zoomed views is now stored in the files.
- The rotation window renders in the background like the main window: rotating and panning never block the GUI, a burst of mouse moves renders only the newest orientation, large picks are previewed with a subset of localizations while dragging (sized by the localizations in view, not by the loaded total) and sharpened as soon as the drag pauses, and the renders use the GPU when it is enabled.
- 3D animations are built in the background at a resolution of your choice (`Resolution (px)` in the animation dialog, independent of the window's size, e.g. 1920 x 1080), with a cancel button; failures are reported instead of silently producing nothing; the frames use the GPU when it is enabled.
- Localizations are standardized to float32 floating-point columns and a uint32 `frame` on loading, saving and in the processing functions (`picasso.lib.standardize_dtypes`, via `ensure_sanity`), the dtypes Localize writes. Pipelines that had promoted columns to float64 (undrifting, z fitting, imports) now produce files and DataFrames half the size, and the GPU renderer no longer converts such columns on every render.
- New `picasso.render.backend` seam: `SplatBackend` (CPU and wgpu implementations), `SplatBackendError`, `describe_active()`, `release_uploads()`, `gpu_settings()`; `picasso.render.render` itself always uses the CPU kernels, `render_scene` goes through the selected backend.
- `picasso.render.render` and `render_scene` accept `max_blur_width`, `global_precision` (the blur of *Global loc. prec.*) and `indices` (per-channel row selections rendered without copying the DataFrame).
- `picasso.render.build_animation` gained a `cancel` callback and returns whether the build completed.
- New `picasso.spatial_index` functions: `query_rect`, `query_circle` (circular picks through the pyramid), `save_render_index`, `read_render_index`, `validate_render_index`, `load_render_index`. `picasso.postprocess.picked_locs` and the functions taking `index_blocks` (`pick_similar`, `combine_locs_in_picks`, `remove_locs_in_picks`, `align_from_picked`, `undrift_from_fiducials`) accept a `spatial_index.RenderIndexPyramid` in that argument.
- `picasso.io.save_locs` gained `render_index`; `picasso.lib.standardize_dtypes` is new.

## 0.11.4
- Plot profile (rectangular) allows for lower bin size.

## 0.11.3
- **Breaking change:** dark times are now the number of frames without signal between binding events, i.e., one frame shorter than before (two binding events in consecutive frames have a dark time of 0). Together with the bright time, each now counts the frames spent in its state. This affects Render's pick info, qPAINT (`picasso.postprocess.evaluate_picks`, `pick_kinetics` and `pick_properties`) and `picasso dark`; influx rates calibrated with earlier versions should be recalibrated. See the [documentation](https://picassosr.readthedocs.io/en/latest/files.html#hdf5-pick-property-files) for the convention.

- Localize: auto contrast button in the bottom right corner of the window; contrast dialog still exists for a numerical input.
- Localize: new `Identify on` mode for multichannel and split-FOV data, *Sum of unregistered channels*. It adds the channels up (in photons) pixel for pixel, without any registration. See the [documentation](https://picassosr.readthedocs.io/en/latest/localize/spline.html#localize-summing-without-registration). The existing mode is renamed *Sum of registered channels*.
- Localize: the channel sum has its own box size, min. net gradient and identification filters, the same whichever channel is displayed. Each channel keeps its own settings for identifying the channels separately (and for registering them for the sum).
- Z binning for 3D calibration (Gaussian astigmatism and spline).
- New `picasso.localize.localize_frames`: a GUI-free wrapper that runs the existing identification and fit on an in-memory frame stack (instead of a movie read from disk) and assigns absolute frame indices, so batched or live input concatenates into one growing localization table. Results are numerically identical to `picasso.localize.localize` on the same frames and parameters.
- Fixed Localize not saying why the channel sum was not shown (channels not registered/aligned yet).
- Render: NeNA is saved to metadata when calculated.
- Fixed saving rotated localizations [#708](https://github.com/jungmannlab/picasso/issues/708)
- Fixed .csv export in plot profile [#709](https://github.com/jungmannlab/picasso/issues/709)
- Fixed Render's 3D slicer failing to open for some z ranges and for data thinner than one slice [#710](https://github.com/jungmannlab/picasso/issues/710).
- Fixed SPINNA single simulation without loading experimental data.

## 0.11.2
- Fixed the calibrations stored in the camera config (z, experimental PSF and sCMOS) not being cleared when switching to a camera the config has no entry for.
- Render saves the color bar (LUT) of the rendered property next to every image exported while rendering by property (`*_colorbar.png`), from the main and from the 3D window.
- Fixed AIM's second round barely correcting the drift ([HohlbeinLab/webSMLM#9](https://github.com/HohlbeinLab/webSMLM/issues/9), reported by @HazenBabcock against a port of this code). Also, thanks to @jhohlbein for the information.
- Fixed AIM drift correction underestimating the drift.
- AIM no longer extrapolates the drift with a cubic spline before the first and after the last segment midpoint, where it could swing far off; the spline knots are now padded as in the reference implementation.
- Fixed Render failing to start when `~/.picasso/settings.yaml` holds a `Render` section without a `Colormap` entry.
- Fixed 3D animations losing full turns
- 3D animations no longer render each position twice, which held one frame at every position of the sequence.
- Fixed error when adding new channels when 3D render is open.
- Updated readme.txt for both installers regarding camera config and plugins.

## 0.11.1

### General
- Plugins can now extend Picasso's [Python API and command line](https://picassosr.readthedocs.io/en/latest/plugins.html#for-developers), not only the GUIs, and can be [installed and enabled without ever opening one](https://picassosr.readthedocs.io/en/latest/plugins.html#managing-plugins-without-a-gui) using the new [`picasso plugins`](https://picassosr.readthedocs.io/en/latest/cmd.html#plugins) command.
- Errors are no longer silently swallowed in the one-click installers. Picasso now redirects errors to a log file (`~/.picasso/logs/picasso.log`), logs every uncaught exception (main thread, worker threads and unraisable ones) there, and shows it in a message box - whose *Show Details...* holds the full traceback.

### Localize
- Localize can fit multichannel data one channel at a time, see the [documentation](https://picassosr.readthedocs.io/en/latest/localize/multichannel.html#localize-analyzing-each-channel).
- A lateral correction (astigmatism / chromatic) can now be kept in its own file and loaded separately from the 3D calibration, see the [documentation](https://picassosr.readthedocs.io/en/latest/localize/3d-calibration.html#localize-appending-or-loading-separately).
- New translation transform model (2 DOF, at least 1 bead pair), offered wherever Picasso fits a geometric transform. It fits a pure shift in x and y.
- Localize loads z-stack `.nd2` files.
- Localize's contrast is not set to auto on opening a new movie.
- ROI id is saved in single-channel localization (if ROIs are used).
- Fixed empty space in Localize's Parameters dialog.
- Fixed drawing ROIs in Localize while the temporal median filter is on.

### Render
- New pick shape in Render: **Box**, an axis-aligned rectangle dragged out to any size with the left mouse button. See the [documentation](https://picassosr.readthedocs.io/en/latest/render/picking.html#render-picking).
- New pick shape in Render: **Brush**, painted freehand with the left mouse button. See the [documentation](https://picassosr.readthedocs.io/en/latest/render/picking.html#render-picking).
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
- Added support for MicroManager "separate image files" acquisitions (one `img_*.tif` per frame in a folder), see [Localize documentation](https://picassosr.readthedocs.io/en/latest/localize/identification.html#localize-extra-features).
- TIFF movies found across several folders can be opened as one concatenated movie (`File` > `Concatenate movies`), with the file order shown for confirmation before loading, see [Localize documentation](https://picassosr.readthedocs.io/en/latest/localize/identification.html#localize-extra-features).
- New fitting model: **Experimental PSF (cubic spline)** — fits an experimentally measured PSF (a cubic-spline model built from a bead z-stack), via the new `picasso.fitting.splinefit_cuda` module on the GPU or `picasso.fitting.splinefit` on the CPU. The spline coefficients of the calibration are computed in pure Python (NumPy/SciPy) on the CPU. In single-channel data, the bead alignment follows the workflow from [Li, et al, Nature Methods, 2018](https://www.nature.com/articles/nmeth.4661). See the [experimental PSF (cubic-spline) fitting documentation](https://picassosr.readthedocs.io/en/latest/localize/spline.html#localize-spline) for details. *Note this is an experimental feature, do let us know if you find any bugs/unexpected behavior*.
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
- `Pick similar` now works with square and rectangular picks, not only circular ones, see [documentation](https://picassosr.readthedocs.io/en/latest/render/menu-tools.html#render-pick-similar).
- Rendering rotated Gaussians.
- More user-friendly measure tool.
- Faster AIM through smarter implementation.
- Localization files now load on a background thread, so the Render window (including the file dialogs) stays responsive while files are read; a progress dialog with a `Cancel` button is shown.
- Faster and more memory efficient (especially for large datasets) SMLM clusterer + progress bar.
- Progress bar for finding cluster centers.
- SMAP localization file reading, see "Other improvements" below.
- Rotation dialog allows for rotations around the localizations or the world (see [3D documentation](https://picassosr.readthedocs.io/en/latest/render/3d.html#render-3d)).
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
- G5M documentation has been updated to include more troubleshooting tips and common issues, see [here](https://picassosr.readthedocs.io/en/latest/render/analysis.html#render-g5m).
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

- **Algorithm for molecular mapping introduced (G5M)**, see documentation [here](https://picassosr.readthedocs.io/en/latest/render/analysis.html#render-g5m). DOI: [10.1038/s41467-026-70198-5](https://doi.org/10.1038/s41467-026-70198-5).
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
