Identification and Fitting
==========================

This page describes the basic workflow of Picasso Localize: identifying single-molecule spots in a movie and fitting them, the optional identification filters, regions of interest and further file actions.

.. _localize-identification:

Identification and fitting of single-molecule spots
---------------------------------------------------

1. **Open a movie.** In ``Picasso: Localize``, drag the file into the window or select ``File`` > ``Open movie``. See :ref:`localize-file-formats` for the supported formats.

   - If the movie is split into multiple μManager ``.tif`` files, open only the first file. Picasso will automatically detect the remaining files according to their file names.
   - Similarly, for consecutive ``.stk`` files (e.g. ``name_001.stk``, ``name_002.stk``, …), open the first file of the desired range and Picasso will automatically include all subsequent files with a higher numeric suffix.
   - When opening a ``.raw`` file, a dialog will appear for file specifications.
   - An IMS file should be displayed immediately in the Localize window. When opening an IMS file with multiple channels, a dialog window will appear allowing you to select the channel that should be loaded.

   You can navigate through the file using the arrow keys on your keyboard. The current frame is displayed in the lower right corner.

2. **Adjust the image contrast** so that the single-molecule spots are clearly visible.

   - The quickest way is the two-handle slider at the bottom of the window, below the frame slider.
   - Alternatively, select ``View`` > ``Contrast`` to type in the black and white points, or to re-enable ``Auto``, which re-scales every frame to its own minimum and maximum.

3. **Open the** ``Parameters`` **dialog** (select ``Analyze`` > ``Parameters``) to adjust spot identification and fit parameters.

4. **Set the identification parameters** in the ``Identification`` group.

   - Set the ``Box side length`` to the rounded integer value of 6 × σ + 1, where σ is the standard deviation of the PSF. In an optimized microscope setup, σ is below one pixel (roughly 0.9 pixels) and the respective ``Box side length`` should be set to 7.
   - The value of ``Min. net gradient`` specifies a minimum threshold above which spots should be considered for fitting. The net gradient sums, over the box, how steeply the intensity rises towards the spot's center. It is proportional to the spot's brightness above the background (in camera counts) and also grows with sharper spots and larger boxes.
   - By checking ``Preview``, the spots identified with the current settings will be marked in the displayed frame. Adjust ``Min. net gradient`` to a value at which only spots are detected (no background).
   - Alternatively, select ``B-spline wavelet`` as the ``Method`` to identify spots by wavelet segmentation, with a threshold in units of the noise instead of the net gradient; see :ref:`localize-wavelet`.

5. (Optional) Tick ``Temporal median filter`` in the ``Identification`` group to subtract a rolling per-pixel background before spots are identified; see :ref:`localize-temporal-median-filter`.

6. (Optional) Set ``Gaussian filter sigma`` in the ``Identification`` group to smooth every frame before spots are identified, which helps when spots are not Gaussian-shaped; see :ref:`localize-gaussian-filter`.

7. (Optional) Restrict the analysis to one or more regions of interest (ROIs) instead of the whole frame; see :ref:`localize-rois`.

8. **Set the photon conversion** in the ``Photon conversion`` group: adjust ``EM Gain``, ``Baseline`` and ``Sensitivity`` according to your camera specifications and the experimental conditions.

   - Set ``EM Gain`` to 1 for conventional output amplification.
   - ``Baseline`` is the average dark camera count.
   - ``Sensitivity`` is the conversion factor (electrons per analog-to-digital (A/D) count).

   These parameters convert camera counts to photons. The fitted positions may depend on them, but the reported photon counts, background and localization precisions depend directly on them. The maximum-likelihood fit also assumes Poisson photon noise, so it works best with correct absolute photon counts.

   - For simulated data, generated with ``Picasso: Simulate``, set the parameters as follows: ``EM Gain`` = 1, ``Baseline`` = 0, ``Sensitivity`` = 1.
   - If you use an sCMOS camera, consider loading a per-pixel camera calibration instead of relying on the two scalars; see :ref:`localize-scmos-calibration`.

   Camera parameters can also be stored and selected automatically through a camera config file; see :ref:`localize-camera-config`.

9. **Run the analysis.** From the menu bar, select ``Analyze`` > ``Localize (Identify & Fit)`` to start spot identification and fitting in all movie frames. The status of this computation is displayed in the window's status bar.

   - After completion, the fit results will be saved in a new file in the same folder as the movie, in which the filename is the base name of the movie file with the extension ``_locs.hdf5``.
   - Furthermore, information about the movie and analysis procedure will be saved in an accompanying file with the extension ``_locs.yaml``; this file can be inspected using a text editor.

Hovering the mouse cursor over a fit marker or over an identification box shows a tooltip listing the properties of that localization — all columns produced by the fit (e.g. ``x``, ``y``, ``photons``, ``bg``, ``sx``, ``sy``).

.. _localize-remembered-parameters:

Remembered parameters
~~~~~~~~~~~~~~~~~~~~~

The following values entered in the ``Parameters`` dialog are all remembered across sessions, under the ``Localize`` section of ``~/.picasso/settings.yaml`` (see :ref:`user-settings-file`):

- ``Box side length``, the identification ``Method``, ``Min. net gradient`` and the wavelet settings,
- the temporal median and Gaussian filters (and whether each is ticked),
- the fit **Model**, **Optimizer** and **Fit mode**.

The same section also stores the last directory used in the file dialogs (``PWD``) and which columns are ticked in the ``File`` > ``Select columns to save...`` dialog when saving fit results.

.. _localize-temporal-median-filter:

Temporal median filter
----------------------

Fluorescence movies often sit on an uneven background: out-of-focus haze, autofluorescent structures or a non-uniform illumination profile. Because a given pixel contains a blinking emitter only for a small fraction of the movie, the *median* of that pixel over a window of frames is a good estimate of its background.

Ticking ``Temporal median filter`` in the ``Identification`` group subtracts that estimate from every frame (clipped at zero) before spots are identified. This removes both the uneven background and any static structure, and should make the detection less sensitive to where in the field of view a spot sits.

``Window (frames)`` sets how many frames go into the median. The default of 51 is a good starting point: it has to be long enough that a given emitter is dark for most of the window (otherwise the emitter ends up in its own background estimate) but short enough to follow slow drifts in the background.

Two things are worth keeping in mind:

- **The filter applies to identification only.** Spots are always cut out of, and fitted on, the *raw* movie, so photon counts, background estimates and the reported localization precisions are unaffected. It changes which spots are found, not how well they are localized.
- **The detection threshold needs re-tuning.** For the net gradient, subtracting the background removes its contribution to the local gradients. For wavelet identification, the noise level the threshold is measured against drops, especially with the default noise estimate from the frame's standard deviation. Turn on ``Preview`` and sweep the value again — while the filter is active the displayed frame is the filtered one, so what you see is what the spot detection sees.



.. important::

   The filter is deliberately **not** applied when calibrating a 3D or an experimental (cubic-spline) PSF: beads in a calibration stack are static and do not blink, so a temporal median would subtract the beads themselves.

For a description of temporal median filtering in the wider context of SMLM analysis, see Martens KJA, Turkowyd B, Endesfelder U, `Raw data to results: a hands-on introduction and overview of computational analysis for single-molecule localization microscopy <https://doi.org/10.3389/fbinf.2021.817254>`_, *Frontiers in Bioinformatics* 1, 817254 (2022).

.. _localize-gaussian-filter:

Gaussian filter
---------------

Spot identification looks for a *single* local maximum per spot. A non-Gaussian point spread function may break up into several local maxima, making identification challenging.

Setting ``Gaussian filter sigma`` in the ``Identification`` group smooths each frame with a Gaussian of that standard deviation (in camera pixels) before spots are identified, which merges those maxima back into one.

A sigma of 0 (the default, shown as ``Off``) disables the filter. Enable ``Preview`` and adapt the value of the filter until the expected regions are outlined successfully.

The same two caveats as for the temporal median filter apply:

- **The filter applies to identification only.** Spots are always cut out of, and fitted on, the *raw* movie, so photon counts, background estimates and the reported localization precisions are unaffected. It changes which spots are found, not how well they are localized.
- **The detection threshold needs re-tuning.** Turn on ``Preview`` and sweep the value again — while the filter is active the displayed frame is the smoothed one, so what you see is what the spot detection sees.

The two filters can be used together: the temporal median background is subtracted first, and the result is then smoothed.

Unlike the temporal median filter, the Gaussian filter *is applied when calibrating* a 3D or an experimental (cubic-spline) PSF — smoothing does not erase static beads, and defocused beads are exactly the kind of multi-peaked PSF the filter helps with.

.. _localize-wavelet:

B-spline wavelet identification
-------------------------------

As an alternative to the net gradient, spots can be identified by the B-spline wavelet segmentation of Izeddin et al. (Izeddin I, Boulanger J, Racine V, Specht CG, Kechkar A, Nair D, Triller A, Choquet D, Dahan M, Sibarita JB, `Wavelet analysis for single molecule localization microscopy <https://doi.org/10.1364/OE.20.002081>`_, *Optics Express* 20, 2081–2095 (2012)). Please refer to the paper for the explanation of the mechanism. It can separate spots closer together than net gradient thresholding and deals with background better.

Select ``B-spline wavelet`` as the ``Method`` in the ``Identification`` group.

Each frame is decomposed with the undecimated wavelet transform, using a third-order B-spline. The second wavelet plane keeps structures of about the size of a diffraction-limited spot, while the pixel noise and the background end up in the other planes. Then:

1. the second plane is thresholded,
2. a watershed splits regions that contain several overlapping spots,
3. regions smaller than ``Min. region area`` are discarded,
4. the centroid of each remaining region, rounded to the nearest pixel, becomes the center of the box that is fitted.

The spots are then localized by the selected fit, exactly as after the net gradient identification.

The settings are:

``Wavelet threshold``
   The threshold on the second wavelet plane, in units of the standard deviation of the noise.

   - Izeddin et al. use values between 0.5 and 2; the default of 0.5 is the value used for the paper's figures.
   - Higher values keep fewer, brighter spots.
   - The threshold is relative to the noise, so it does not depend on the camera gain or the brightness of the dye.
   - Check with ``Preview`` to select the right value.

``Noise estimate``
   How the noise is estimated in each frame.

   - ``Image std`` (the default, as in the paper) takes the standard deviation of the frame, which is a good estimate when spots are sparse.
   - ``First wavelet plane (MAD)`` takes the median absolute deviation of the finest wavelet plane (Donoho and Johnstone, 1995). It stays accurate in frames crowded with spots or with an uneven background, where the standard deviation of the frame overestimates the noise and fewer spots are found.

``Min. region area``
   Regions of fewer pixels are removed as noise; 4 as in the paper.

Things to keep in mind:

- **No net gradient.** Spots found by wavelet segmentation have no net gradient, so neither the identifications nor the localizations fitted from them have a ``net_gradient`` column. When localizations with and without the column are combined (e.g., with ``picasso join``), the column is dropped with a warning.
- **The box.** ``Box side length`` still sets the size of the fitted box, and spots whose box does not fit into the frame are skipped. Unlike the net gradient identification, which finds at most one spot per box, the watershed can separate spots that are closer than one box.
- **Filters and ROIs** work as for the net gradient identification. With ROIs, each ROI (plus a margin) is segmented on its own, including the noise estimate.
- **Calibrations** — 3D and spline PSF calibration, lateral calibration and channel registration — detect the beads with the selected method, too.
- The border of a frame is extended by mirroring, a detail the paper does not specify.


.. _localize-rois:

Regions of interest (ROIs)
--------------------------

By default, Picasso analyzes the whole frame. If you are only interested in certain parts of the movie, you can restrict the analysis to one or more rectangular regions of interest (ROIs). Spots outside the ROIs are ignored, which also speeds up the analysis. There are two ways to work with ROIs:

- **With the mouse, directly on the image.** Drag a rectangle with the left mouse button to add a ROI; repeat to add as many as you like. To remove a ROI, double-click inside it. ROIs are outlined in blue, and the one currently selected is highlighted in cyan.

- **Numerically, in the Parameters dialog.** Open ``Analyze`` > ``Parameters``. The ``ROIs`` field in the ``Identification`` group summarizes the current selection:

  - empty (``Whole frame``) means the entire frame is analyzed,
  - a single ROI is shown as its four coordinates ``y_min, x_min, y_max, x_max`` (in camera pixels), which you can edit directly in the field,
  - several ROIs are shown as a count (e.g. ``3 ROIs``).

  Click ``Edit ROIs...`` to open a small dialog where you can add, edit, remove, or clear all ROIs in a table.

To go back to analyzing the whole frame, simply remove all ROIs (double-click them, empty the single-ROI field, or use ``Clear`` in the ``Edit ROIs...`` dialog).

If ROIs overlap, Picasso automatically trims them so that no spot is detected twice, so you do not need to draw them precisely. As with the rest of the identification settings, turn on ``Preview`` to check which spots fall inside your ROIs before running the full analysis.

.. _localize-roi-id:

The ``roi_id`` column
~~~~~~~~~~~~~~~~~~~~~

All ROIs are fitted together into one localization file. To keep them apart afterwards, the saved file carries an extra ``roi_id`` column holding the index of the ROI each localization was found in:

- 0 for the first ROI, 1 for the second, and so on, in the order the ``ROI`` entry of the metadata lists them;
- -1 marks a localization inside none of them, which a fitted position right at an ROI's edge can be.

The ids are computed before drift correction, so they still name the ROI a localization was detected in even after the coordinates have been shifted.

Split-FOV mode (``Regions = channels``) adds no ``roi_id``: there each region is a channel of its own and is saved to its own file anyway.

.. _localize-extra-features:

Extra features
--------------

The ``File`` menu offers several further ways to open movies and to save or load identifications.

.. _localize-opening-channels:

Opening several channels
~~~~~~~~~~~~~~~~~~~~~~~~

``File`` > ``Open one multichannel movie``
   Opens a single multichannel file (``.ims``, ``.czi``, ``.lif`` or ``.nd2``) and loads **every** channel at once, one per channel, rather than prompting for a single channel to load.

``File`` > ``Open channels from several movies``
   Opens several separate movie files and loads each as one channel. The channel name is taken from the file's metadata where available, otherwise from the file name.

When more than one channel is loaded (by either of the two actions above), a channel selector appears below the image so you can switch between channels; identification, fitting and saving then operate on the currently active channel. Up and down arrow keys can be used to navigate across the channels.

.. _localize-micromanager-folder:

Open MicroManager image folder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``File`` > ``Open MicroManager image folder`` opens a MicroManager acquisition that was saved as **separate image files** rather than as a single multi-page stack: one single-page ``img_*.tif`` per frame in a folder, e.g. ``img_channel000_position000_time000000000_z000.tif`` in MicroManager 2.0 or ``img_000000000_Default_000.tif`` in MicroManager 1.4.

- Select the acquisition folder and Picasso assembles the whole sequence into one movie, ordered by frame index.
- Channel, position and z are held fixed at the first frame's values, so a multi-channel or multi-position acquisition is **not** interleaved into a single movie.
- Only the first frame is read when the movie is opened (the rest are read on demand during localization), so even acquisitions of tens of thousands of files open quickly.

.. _localize-concatenate-movies:

Concatenate movies
~~~~~~~~~~~~~~~~~~

``File`` > ``Concatenate movies`` opens an acquisition that was split into several TIFF files, possibly spread over different folders, as a **single** movie whose frames run through the files one after another.

1. Select a folder. Picasso searches it and all of its sub-folders for TIFF movies.
2. Picasso shows the list in the order the frames will be concatenated: sorted by folder and file name, with numbers compared numerically, so ``run_2`` comes before ``run_10``.
3. Check that order before continuing. You can drag entries to reorder them, use ``Move up`` / ``Move down``, ``Remove`` files that do not belong, and ``Add files...`` from folders the search did not cover.

Each entry is one whole movie: the continuation files of a split OME-TIFF stack (``*_1.ome.tif``, ...) and the individual frames of a MicroManager "separate image files" folder are collapsed into their parent movie, so no frames are repeated.

All files must have the same frame size and data type; if one does not, Picasso names it and the movie is not opened.

The metadata is taken from the first file, with the frame count set to the total. The concatenated file paths and their frame counts are stored in the localization metadata (``Concatenated Files`` and ``Frames per File``), so it stays traceable which frame range came from which file.

.. _localize-saving-loading-identifications:

Saving and loading identifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``File`` > ``Save identifications``
   Saves the current set of identifications (frame, x, y, net gradient and identification id, where applicable; wavelet identifications have no net gradient) to an HDF5 file with a companion YAML metadata file. By default the suggested filename is ``<movie_base>_identifications.hdf5``.

   The accompanying YAML stores the original movie metadata together with the parameters used at the time of saving, so they can be restored when the identifications are loaded again:

   - the ``Box Size`` and the ``Identification Method``,
   - its threshold (``Min. Net Gradient``, or the ``Wavelet Threshold``, ``Wavelet Noise Estimate`` and ``Wavelet Min. Area``),
   - ``Temporal Median Window`` and ``Gaussian Filter Sigma``.

``File`` > ``Load identifications``
   Loads identifications previously saved with ``Save identifications``. The identifications are clipped to the current movie's bounds (using the current ``Box Size``), and the identification parameters stored in the YAML sidecar (``Box Size``, the identification method and its threshold, ``Temporal Median Window``, ``Gaussian Filter Sigma``) are restored.

``File`` > ``Load picks as identifications``
   Allows the user to load circular picks (from Picasso Render) as identifications. Additionally, the drift correction file (.txt) can be loaded to adjust the positions of the identifications throughout acquisition.
   
   The current box size will be used to make the identification, however, min. net gradient will **not** be applied to the identifications.
   
   This can be used to backtrack the raw signal from given pick regions.

``File`` > ``Load locs as identifications``
   Similar to loading picks as identifications (see above) but uses localizations as input.

   The user is asked to provide the number of frames around localizations to be used for the identifications, i.e., how many frames before and after the frame of the localization should be included in the identifications.

   For each localization, 2 * n_frames + 1 identifications will be assigned, thus if localizations are close together the identifications may overlap.

   This can be used to backtrack the raw signal from given localizations.

.. important::

   For all three loading actions above, changing any identification parameter (box size, min. net gradient, etc.) will reset the loaded identifications. Use ``Analyze`` > ``Fit``, rather than ``Analyze`` > ``Localize (Identify & Fit)``, to fit the loaded identifications without resetting them.

.. _localize-save-spots:

Save spots
~~~~~~~~~~

``File`` > ``Save spots`` cuts out and saves the identified spots (NxBxB array, with N spots and B being the box side length). The spots can be saved as a ``.npy`` file or as a ``.tif`` file.

.. _localize-background-loading:

Background loading
~~~~~~~~~~~~~~~~~~

Loading runs in the background, so the window stays responsive while the files are read, and a progress dialog with a ``Cancel`` button is shown.

- Canceling stops before the next file begins (a file already being read is finished first).
- This also applies to opening a single movie.
