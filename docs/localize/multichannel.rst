.. _localize-multichannel:

Multichannel Fitting
====================

Picasso offers three ways to fit data with several channels, either loaded as separate movies or imaged side by side on one sensor (split field of view):

- **Multichannel 2D Gaussian fitting** — the channels are fitted jointly with a spherical Gaussian, sharing one position and width per molecule; needs only a :ref:`channel registration <localize-channel-registration>`.
- **Multichannel spline PSF** — the channels are fitted jointly with one measured PSF per channel; see :ref:`localize-multichannel-spline`.
- **Each channel on its own** — every channel is fitted independently, with any model and without a registration.

.. _localize-channel-registration:

.. admonition:: Channel registration

   A channel registration is one coordinate transform per channel that maps a
   position in the reference (first) channel to where the same molecule
   appears in that channel. It is measured from spots seen in every channel
   (beads or the blinking signal itself) and is either part of a multichannel
   spline calibration (see :ref:`localize-multichannel-spline`) or saved on its
   own as a ``.yaml`` for the multichannel 2D Gaussian fit (see
   :ref:`localize-registering-channels`). The available transform models are
   described in :ref:`localize-lateral-transform-models`.

By default (``Identify on`` > ``Each channel separately``), the spots are identified in every channel on its own, with that channel's identification settings. The joint fits then pair the spots across the channels through the registration and fit only those found in *every* channel; fitting each channel on its own fits all of them. When one channel is dimmer, so that many of its spots go undetected, identify on the sum of the channels instead; see :ref:`localize-identify-on-sum`.

.. _localize-multichannel-gaussian:

Multichannel 2D Gaussian fitting
--------------------------------

Several spatially-registered channels can also be fitted jointly with a **spherical Gaussian**, sharing one ``x``, ``y`` and width per molecule.

This is the same global-fitting idea as the :ref:`localize-multichannel-spline` (globLoc, `Li et al., Nature Communications, 2022 <https://doi.org/10.1038/s41467-022-30719-4>`_), but it needs **no measured PSF** — only a :ref:`channel registration <localize-channel-registration>`.

It is available for the ``2D spherical Gaussian`` model only.

.. note::

   This feature is experimental — please report any unexpected behavior on our `GitHub issues page <https://github.com/jungmannlab/picasso/issues>`_.

.. _localize-registering-channels:

Registering the channels
~~~~~~~~~~~~~~~~~~~~~~~~

Load the channels first, in either of the two layouts:

- **Separate movies** — ``File`` > ``Open channels from several movies``, or ``Open one multichannel movie`` for a single file holding several. **The first channel loaded is the reference channel.**
- **Split field of view** — if the channels are imaged side by side on one sensor, load the single movie, tick **Regions = channels** in the ``Parameters`` dialog and drag one ROI onto each channel. **The first region is the reference channel**; all further regions have the same size.

Either way the localizations come out in the reference channel's coordinates.

Then build a registration with ``Calibration`` > ``Register channels (2D)``, which offers two ways to measure it.

.. tip::

   **Beads are the recommended way to measure the registration.** They are bright, static and present in every channel, so the correspondences are unambiguous and the transform is fitted from far more pairs than blinking data provides. The registration is best when:

   - **several fields of view are imaged with sparse beads** — sparse, so that neighboring beads cannot be mismatched, and several fields, so that the pairs cover the whole sensor;
   - the bead stack is acquired **on the same day as the measurement**, ideally directly before or after it to minimize the effect of drift.

**From bead data...**
   Measure the registration from the movie(s) currently open, so load the bead images themselves as the channels: one bead movie per channel, reference first (in split-FOV mode, the single bead movie holding every region).

   - Beads are detected and fitted with the current ``Box side length`` and ``Min. net gradient``, matched to the reference channel's beads, and a transform is fitted per channel with outlier pairs dropped.
   - Tick **Each frame is a different field of view** when the bead movie scans several stage positions rather than repeating one field.

     - Beads are then detected frame by frame and **only ever paired with beads in the same frame** — every field images onto the same sensor coordinates, so pooling them would pair beads that are nowhere near each other — while every field's pairs constrain the one transform.
     - More fields means more correspondences and a better-conditioned fit, which matters most for the flexible models.
   - Left unticked, the frames are treated as repeats of a single field and averaged to beat down the noise, which is what a plain bead acquisition wants.

**From current signal...**
   The fallback when no bead data was acquired: measure the registration from the loaded movies themselves, with no extra acquisition.

   - The channels are frame-synchronized, so the same molecule blinks in every channel in the *same* frame; pairing those detections frame by frame pins down the inter-channel transform.
   - A dialog asks for the frame window and how many frames are evenly sampled from it, and the detection uses the current identification settings. **Use a high** ``Min. net gradient`` **so only bright, unambiguous spots are paired.**
   - This route both **builds** a registration from scratch and **re-aligns** one that has drifted: when a registration is already loaded it seeds the pairing, otherwise the pairing is bootstrapped from the data alone.

``Transform model`` chooses how the channels are related — ``translation`` (a pure xy shift), ``affine`` (the default), ``projective``, ``polynomial2`` or ``polynomial3`` — with the same trade-offs as described under :ref:`localize-lateral-transform-models`.

The registration is saved as a small ``.yaml`` (by default ``<movie>_channel_reg.yaml``) and loaded straight away. Picasso reports, per channel, how many correspondences were paired and the residual RMS in camera pixels — check those before fitting: a registration built from too few pairs, or with an RMS approaching a pixel, will hold the channels together at the wrong place.

.. _localize-multichannel-gaussian-fitting:

Fitting
~~~~~~~

1. Open ``Analyze`` > ``Parameters`` and set **Model** to ``2D spherical Gaussian``.
2. In the **Multichannel: channel registration** box, click ``Load registration`` and choose the ``.yaml`` (``Clear`` drops it again).
3. Choose the **Optimizer** — ``Least squares`` or ``MLE``.
4. Decide whether to link the photon counts (see :ref:`localize-linking-photon-counts`).
5. Tick **Use GPU** to fit on the GPU; leave it unticked for the CPU. The check box is only visible if a CUDA-capable GPU is present and the required numba package is installed (see :doc:`/getting-started/installation`).
6. Identify the channels, then run ``Analyze`` > ``Fit`` (or ``Localize (Identify & Fit)``).

Only molecules detected in *every* channel are fitted, so identify each channel first:

- With several channels loaded, ``Identify`` (:kbd:`Ctrl+I`) analyzes all of them in turn.
- In split-FOV mode the whole movie is identified at once and the detections are split by region, so nothing extra is needed; they are confined to the reference region automatically, and one localization comes out per molecule.
- When one channel is much dimmer than the others, identify on the channel sum instead; see :ref:`localize-identify-on-sum`, which works from a loaded channel registration exactly as it does from a spline calibration.

**With no registration loaded, nothing changes:** the spherical Gaussian fits the active channel alone, as before. The joint fit runs only when a registration is loaded *and* the data actually has several channels — either several movies open, or ``Regions = channels`` with a split-FOV registration.

To fit every channel on its own instead — with any model, and without a registration — set ``Fit`` to ``Each channel separately``; see :ref:`localize-analyzing-each-channel` below.

.. _localize-linking-photon-counts:

Linking the photon counts
~~~~~~~~~~~~~~~~~~~~~~~~~

Link photon counts across channels is off by default.

A multichannel spline calibration carries one measured PSF per channel, and with it that channel's own brightness, so linking there means "one molecule's total emission, split as the calibration says".

A channel registration carries no such brightness information — only geometry. Linking would therefore mean *the same photon count in every channel*, which is right for equally split, redundant channels and wrong for an uneven beam splitter or for channels of different spectral throughput.

So, unless the channels are known to be balanced:

- **Unlinked (default)** — each channel fits its own photon count and background; only ``x``, ``y`` and the width are shared. Adds per-channel columns, one set per channel ``c``:

  - ``photons_ch<c>`` and ``bg_ch<c>`` — that channel's photon count and background,
  - ``rel_photons_ch<c>`` — that channel's share of the total photons, so the values sum to 1 per localization.

  Supported for 2 to 6 channels.

- **Linked** — one photon count and background shared across the channels.

In both modes ``photons`` and ``bg`` are the **totals across all channels**, so the two are directly comparable with each other and with the spline fit.

.. _localize-analyzing-each-channel:

Analyzing each channel on its own
---------------------------------

The two multichannel fits above tie the channels together:

- they need a registration (or a measured multichannel PSF),
- they fit one shared position per molecule,
- they keep only the molecules detected in *every* channel,
- they exist for the spherical Gaussian and the spline PSF only.

This section describes how to fit each channel independently of one another. The ``Fit`` setting in the ``Parameters`` dialog chooses between the two:

**Jointly (registered channels)**
   The default: a loaded multichannel spline calibration or channel registration runs the global fit described above, and with none loaded the active channel is fitted by itself.

**Each channel separately**
   Every loaded channel (or every split-FOV region) is fitted on its own, one after another, with **its own** fitting model, optimizer and calibrations, and each is saved to its own file. No registration is needed, nothing is dropped, and the channels come out independent.

.. note::

   This feature is experimental — please report any unexpected behavior on our `GitHub issues page <https://github.com/jungmannlab/picasso/issues>`_.

.. _localize-each-channel-running:

Running it
~~~~~~~~~~

.. tab-set::

   .. tab-item:: GUI

      1. Load the channels, in either layout:

         - ``File`` > ``Open channels from several movies`` (or ``Open one multichannel movie``), or
         - for channels imaged side by side on one sensor, load the single movie, tick **Regions = channels** in the ``Parameters`` dialog and drag one ROI onto each channel, as described above for multichannel fitting.

      2. Open ``Analyze`` > ``Parameters`` and set **Fit** to ``Each channel separately``.
      3. Set up each channel: select it (the channel selector below the image, or the region in the image) and choose its ``Model``, ``Optimizer``, ``Min. net gradient`` and calibrations. See :ref:`localize-each-channel-settings` below.
      4. Run ``Analyze`` > ``Identify`` (:kbd:`Ctrl+I`), which analyzes every channel in turn, and then ``Analyze`` > ``Fit`` — or ``Localize (Identify & Fit)`` for both at once.
      5. The status bar reports each channel as it is fitted and, at the end, how many spots were fitted in how many channels.

   .. tab-item:: Command line

      Separate movies are already independent runs — ``picasso localize`` processes each file on its own. For a split field of view, add ``--regions-separately`` (``-rs``) to a run with several ``--roi`` regions:

      .. code-block:: bash

         picasso localize movie.tif -b 7 --regions-separately \
             --roi 0 0 256 256 --gradient 5000 --fit-method lq \
             --roi 0 256 256 512 --gradient 2000 --fit-method spline \
             --spline-calibration ref_psf.hdf5 --spline-calibration ch1_psf.hdf5

      - Each region is fitted on its own and written to ``movie_ref_locs.hdf5``, ``movie_ch1_locs.hdf5``, ... in that region's own coordinates.
      - ``--gradient``, ``--fit-method`` and ``--spline-calibration`` may be given once per ``--roi`` (in the same order as the regions) or once for all of them.
      - ``--gradient`` accepts per-region values in an ordinary run too, since regions imaged through different optics need not share a brightness scale.

   .. tab-item:: Python

      - :func:`picasso.localize.fit_independent` fits one set of detections per movie.
      - :func:`picasso.localize.fit_split_fov_independent` fits the regions of one movie, returning each region's localizations in its own coordinates together with its metadata.
      - Both take the fitting method, the convergence settings and the spline calibration either once for all channels or once per channel.
      - :func:`picasso.localize.split_locs_by_region` splits an existing set of localizations by region in the same way.

.. _localize-each-channel-settings:

What each channel carries
~~~~~~~~~~~~~~~~~~~~~~~~~

Fitted separately, a channel is a dataset of its own.

**Separate movies**
   Every channel keeps its own ``Parameters`` settings, which are swapped in when the channel is selected. To use the same values for all channels instead, tick them under ``Same settings across channels`` (shown once several movies are loaded); the current values are then copied to every channel:

   - ``Box size`` (off by default),
   - ``Min. net gradient``, or ``Wavelet settings`` with the wavelet identification; this also shares the identification method (off by default),
   - ``Camera settings``: camera, baseline, EM gain, sensitivity and pixel size (off by default),
   - ``PSF calibration``: the experimental PSF (spline) calibration (**on** by default, which is what the joint fit needs). The PSF is measured per channel, so for fitting each channel separately, untick it and load one calibration per channel.

   The model and optimizer (with the convergence criterion, maximum iterations and ``Use GPU``), the astigmatism z calibration and ``Fit Z``, the lateral corrections and the identification filters (temporal median, Gaussian filter) are always kept per channel. The frame range is always shared.

**Split field of view**
   The regions are channels of one movie, so they share one set of settings, except:

   - ``Min. net gradient``, or the wavelet settings with the wavelet identification, kept per region (select a region and the identification settings tune that region alone).
   - The model and the PSF calibration, kept per region when the regions are fitted separately.

   The ``Edit ROIs...`` table lists each region's model and PSF calibration alongside its coordinates and threshold, so the whole setup can be checked at a glance.

.. important::

   A **multichannel** spline calibration cannot be used here. Picasso refuses it and asks for one single-channel calibration per channel rather than silently fitting every channel with the reference channel's PSF.

.. _localize-each-channel-output:

The resulting file
~~~~~~~~~~~~~~~~~~

One file per channel, saved next to its movie exactly as a single-channel run saves its own:

- **Separate movies** — ``<movie>_locs.hdf5`` per channel; when several channels come from one file, the channel name is appended (``<movie>_<channel>_locs.hdf5``).
- **Split field of view** — ``<movie>_ref_locs.hdf5``, ``<movie>_ch1_locs.hdf5``, ... , one per region, named after the labels drawn beside the regions.

Each split-FOV file holds **that region's own coordinates**:

- The region's top-left corner is subtracted from ``x`` and ``y``, and the metadata gives the region's width and height (with the region's position on the sensor recorded under ``Region``).
- A region is therefore a stand-alone channel, and loading the files side by side in Picasso: Render overlays them, rather than placing them next to each other as they sat on the camera.
- The metadata also records ``Fit mode: Each channel separately`` and which channel or region the file came from, so a file from this mode can always be told from one the joint fit produced.

Everything a single-channel run does afterwards is done per channel: the astigmatic z fit when ``Fit Z`` is ticked, and the drift correction when ``AIM`` or the fiducial-based correction is selected — each channel writing its own ``_locs_undrifted.hdf5`` and drift file.
