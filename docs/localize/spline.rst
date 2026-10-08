.. _localize-spline:

Experimental PSF (Cubic Spline)
===============================

Picasso can fit an **experimentally measured PSF** to every spot. The measured PSF is stored as a cubic spline — a smooth, piecewise-polynomial model built from a bead z-stack — and each spot is fit to that spline.

- This captures aberrations and engineered PSFs (e.g. astigmatism) that a Gaussian cannot describe.
- A 3D calibration recovers the axial position ``z`` directly: a single fit returns ``x``, ``y``, ``z``, photons and background, with no separate astigmatism z-calibration step.
- A 2D calibration models a single focal plane (no ``z``).

Building a calibration follows the scheme of `Gpuspline <https://github.com/gpufit/Gpuspline>`_, its license is reproduced in `LICENSES/Gpuspline-LICENSE.txt <https://github.com/jungmannlab/picasso/blob/master/LICENSES/Gpuspline-LICENSE.txt>`__.

**Localization precision.** The fit returns the fitted parameters but no uncertainties, so Picasso evaluates the Cramer-Rao lower bound separately to fill ``lpx``, ``lpy``, ``lpz``, ``photons_unc`` and ``bg_unc``. GPU with CUDA is used if detected, otherwise the process runs on the CPU.

The method combines three published works:

- the experimental-PSF localization workflow and bead alignment of `Li et al., Nature Methods 15, 367–369 (2018) <https://doi.org/10.1038/nmeth.4661>`_,
- the cubic-spline PSF model for single-molecule data introduced by `Babcock & Zhuang, Scientific Reports 7, 552 (2017) <https://doi.org/10.1038/s41598-017-00622-w>`_,
- the fitting algorithm of `Przybylski et al., Scientific Reports 7, 15722 (2017) <https://doi.org/10.1038/s41598-017-15313-9>`_.

The multichannel variant (see :ref:`localize-multichannel-spline` below) additionally follows the global-fitting approach of globLoc, `Li et al., Nature Communications 13, 3133 (2022) <https://doi.org/10.1038/s41467-022-30719-4>`_.

.. _localize-spline-building:

Building a spline calibration
-----------------------------

A calibration is built from a **bead z-stack**: image a sample of sparse, bright, sub-diffraction beads while scanning the stage through focus in even steps. This is the workflow described in `Li et al., Nature Methods 15, 367–369 (2018) <https://doi.org/10.1038/nmeth.4661>`_, however, PSF scaling was adapted to fit Gpufit's workflow.

.. dropdown:: 1 · Detect the beads
   :icon: search

   The beads are detected once and their positions are reused for every z step.

   - Detection runs on the middle third of the scan, where the beads should be closest to focus and brightest, with the identification settings of the ``Parameters`` dialog (net gradient or wavelet).
   - The temporal median filter is never applied here: the beads are static, so it would subtract them.
   - Each detection is rounded to the pixel grid. Detections closer than one box size are merged into one.
   - Beads whose box would reach past the frame edge are dropped.
   - With several fields of view per z step, beads are detected and merged within each field of view separately.

.. dropdown:: 2 · Cut out a volume per bead
   :icon: package

   For every bead, a box is cut out of every frame and converted to photons with the camera parameters, which gives one ``box × box × z`` volume per bead.

   With ``Z binning``, that many consecutive z steps are then averaged into one slice (see ``Z binning`` in the ``Calibrate spline PSF`` dialog, described in the ``GUI`` tab below).

.. dropdown:: 3 · Register the beads in 3D and reject outliers
   :icon: git-compare

   Beads may reach focus at slightly different stage positions due to coverslip tilt and sit at slightly different sub-pixel positions, so they are aligned to each other before averaging.

   - **Focus.** The sharpest slice of the mean of all beads, the one with the smallest fitted Gaussian width, marks the focus.
   - **Alignment.** Each bead is aligned to a reference by 3D cross-correlation. Only the slices around focus are correlated. The correlation peak is located with sub-voxel precision by upsampling it 20-fold with cubic-spline interpolation, and the bead volume is shifted there by cubic-spline interpolation.
   - **Iteration.** The first reference is the brightest bead at focus. In two further rounds, all beads are realigned to the average of the beads kept so far.
   - **Outliers.** In every round, a bead is rejected if its normalized cross-correlation with the average is unusually low or its mean-square difference from the (brightness-matched) average is unusually high, i.e. more than 2.5 robust standard deviations from the median of all beads. At least half of the beads, and never fewer than three, are always kept. The bead gallery written next to the calibration shows which beads were kept and why the others were rejected (see :ref:`localize-spline-beads`).
   - **Centering.** The average is finally shifted laterally so that its center at focus sits exactly on the box center, where a fitted shift of zero places the emitter.

.. dropdown:: 4 · Smooth along z
   :icon: pulse

   Each pixel's intensity-vs-z profile is smoothed with a cubic smoothing spline, as in Li et al. (2018). The amount of smoothing is chosen automatically by generalized cross-validation, so noise is removed without washing out the axial changes that encode ``z``. At least five z slices (after binning) are needed for this.

.. dropdown:: 5 · Normalize the PSF
   :icon: dash

   - The focus is located again on the smoothed volume.
   - The background, the minimum of the volume, is subtracted, and the volume is divided by the peak of the in-focus slice, so the PSF model has a peak of 1.
   - The sum of the in-focus slice is stored as well; it converts the fitted amplitude into the number of photons.

   Li et al. (2018) instead normalize the in-focus slice to a sum of 1; the unit peak used here keeps the fit's starting values valid.

.. dropdown:: 6 · Compute the spline coefficients
   :icon: graph

   The normalized volume is interpolated by a cubic spline in x, y and z (natural boundary conditions, i.e. zero curvature at the edges of the volume), following the coefficient layout of Gpuspline. The spline passes exactly through every voxel.

   - A ``2D (single plane)`` calibration uses only the in-focus slice.
   - ``z = 0`` is placed at the center of the stage scan, or at the axial intensity peak with ``Set z = 0 at max. intensity`` checked.

.. tab-set::

   .. tab-item:: GUI

      Load the bead movie and select ``Calibration`` > ``Calibrate spline PSF``. A dialog collects:

      **Calibration step size (nm)**
         The axial stage step between consecutive frames (or z-positions).

      **Number of frames per step size** and **Frame order**
         For movies that image several fields of view (FOVs) to collect more beads: the number of FOVs, and whether all FOVs are imaged at each z position before the stage moves (``Different FOVs first``) or each FOV gets its own full z-stack (``Different z positions first``). Each FOV should show different beads.

      **Z binning (steps per bin)** (default 1)
         Averages that many consecutive z-positions into one slice of the PSF model, so the spline's axial knots are *binning × step size* apart (the dialog shows the resulting bin size).

         - Each slice sits at the mean stage position of its steps; trailing steps that do not fill a whole slice are dropped.
         - This can mitigate spikes in the fitted ``z`` positions of single emitters at certain values, since the PSF model is smoother. For an astigmatic PSF, bins of about 50 nm remove these spikes without a measurable loss of precision, and the coarser model also fits faster. *Keep in mind that your system might work better with different binning!*
         - PSFs with fine axial structure (e.g. interference PSFs) need finer bins.
         - The diagnostic plot compares every single-frame bead spot with the stage position of its own frame, so binning does not inflate the reported precision.

      **Spline PSF model**
         ``3D (recovers z)`` or ``2D (single plane)``.

      **Magnification factor** (default 0.79)
         Scales the fitted ``z`` to correct for the refractive-index mismatch, as in the astigmatism fit (`Huang et al., Science 319, 810–813 (2008) <https://doi.org/10.1126/science.1153529>`__). It is stored in the calibration and applied at fit time, not during calibration.

      **Set z = 0 at max. intensity**
         Define ``z = 0`` at the axial intensity peak of the averaged PSF instead of the center of the stage scan. Only meaningful for a PSF with a single, well-defined focus (e.g. astigmatism); off by default. This will impact the effect of the magnification factor if the measured calibration data is offset.

      The box size and minimum net gradient are taken from the main ``Parameters`` dialog. You are then asked where to save the calibration ``.hdf5``. Written next to it are:

      - a **diagnostic plot** (a ``.png`` with the same base name),
      - a **bead gallery** (``<base>_beads.png``), showing which individual beads were averaged into the PSF and which were rejected.

   .. tab-item:: Command line

      .. code-block:: bash

         picasso spline-calibrate my_beads.tif -s 20

      ``-s/--step`` (the z step in nm) is required. Useful options:

      - ``-b`` box side length (default 13),
      - ``-g`` minimum net gradient,
      - ``-m`` model (``spline-3d`` / ``spline-2d``),
      - ``-fps`` / ``-fo`` frames-per-step and order,
      - ``-zb`` z binning,
      - ``-mf`` magnification factor,
      - ``-cz`` to set ``z = 0`` at the intensity peak,
      - the camera parameters ``-bl`` / ``-se`` / ``-ga`` / ``-px`` (baseline, sensitivity, gain, pixel size),
      - ``-o`` for the output path (default ``<movie>_spline_calib.hdf5``).

.. important::

   **The fit box size must not be larger than the box size the calibration was built with.** If it is larger, Picasso Localize shows a dialog and offers to set the box size to the calibration's value (you then re-run identification before fitting).

.. _localize-spline-plot:

Reading the calibration plot
----------------------------

The diagnostic ``.png`` summarizes the averaged PSF and lets you judge the calibration at a glance. Its title reports the number of beads, the z range, the box and pixel size, and — when available — the model-vs-data agreement (median R² and NRMSE). Every image panel shares one intensity scale, and one camera pixel is drawn at the same physical size in all panels.

**xy slices (across z)**
   The PSF seen face-on at evenly spaced z-planes; the in-focus (sharpest) slice is outlined.

**xz and yz cross-sections**
   Side views with z on the vertical axis; a cyan line marks the sharpest slice.

**Axial intensity profile**
   The brightest normalized pixel per slice versus stage position.

For a 3D calibration, Picasso also re-fits the individual beads using the spline model and adds three panels:

**Estimated z vs stage**
   Recovered z against the known stage position, with the identity line. Points should be found around the diagonal across the whole z range.

**Axial bias**
   The mean signed z error per step; ideally flat and near 0 nm.

**Axial precision**
   The spread of the recovered z per step (nm).

.. _localize-spline-beads:

Checking which beads went into the PSF
--------------------------------------

Not every detected bead is averaged into the PSF. While registering the beads, Picasso compares each one against the running average and discards those whose shape disagrees with it — by correlation and by residual — keeping at least half of them.

This aims to remove doublets, aggregates, and beads sitting at a different height, but it is worth looking at: if many beads are dropped, the PSF may genuinely vary across the field of view, and the calibration is then built from a biased subset.

To look at the filtering, click ``Inspect beads...`` in the message shown when a calibration finishes, or use ``Calibration`` > ``Inspect calibration beads``; the same gallery is written next to the calibration as ``<base>_beads.png``.

For multichannel data (see :ref:`localize-multichannel-spline` below), each channel (or split-FOV) calibration builds its own PSF model from its own beads and therefore filters independently: the inspector has a channel selector, and one gallery per channel is saved as ``<base>_ch{c}_beads.png``.

A healthy calibration rejects a few clearly odd beads. Rejected beads that look just like the kept ones — or rejections concentrated in one corner of the field of view — mean the PSF is field-dependent, and a smaller ROI will describe the data better.

.. _localize-spline-fitting:

Fitting with the spline PSF
---------------------------

1. Open ``Analyze`` > ``Parameters`` and set **Model** to ``Experimental PSF (cubic spline)``.
2. In the **Experimental PSF (spline)** box, click ``Load calibration`` and choose your ``.hdf5``. The last-used calibration is remembered between sessions, and calibrations can be loaded automatically per camera and emission wavelength via the ``spline-calibrations`` config field (see :ref:`localize-camera-config-calibrations`).
3. Choose the **Optimizer**: ``Least squares`` or ``MLE`` (Poisson maximum likelihood). ``MLE`` is recommended.
4. Tick **Use GPU** to run the fit on the GPU; leave it unticked to fit on the CPU. The checkbox is only available when a CUDA GPU is detected.
5. Run ``Analyze`` > ``Localize (Identify & Fit)`` (or ``Fit`` for already-identified spots).

In addition to the usual columns, spline fits report:

- per-localization precisions (``lpx``, ``lpy``, and ``lpz`` for 3D, in nm),
- ``photons`` and ``bg`` with their uncertainties (``photons_unc``, ``bg_unc``),
- the fit quality: ``log_likelihood`` for MLE or ``chi_square`` (the sum of squared residuals) for least squares, plus ``reduced_chi_square``, normalized for the box size and the photon counts so it can be compared between spots,
- the number of ``iterations`` each fit took.

A 3D calibration adds the recovered ``z`` (and ``lpz``). The accompanying ``_locs.yaml`` records the spline calibration model and file path used, and which device performed the fit.

.. _localize-multichannel-spline:

Multichannel spline PSF (e.g. biplane)
--------------------------------------

Several spatially-registered channels (e.g. biplane setups) can be fit simultaneously, sharing one ``x``, ``y`` and ``z`` per molecule. (To fit the channels one at a time instead, each with its own PSF, see :ref:`localize-analyzing-each-channel`.) The calibration needs one bead z-stack per channel, all scanned over the same z range with the same number of frames.

This implements the global-fitting (globLoc) approach of `Li et al., Nature Communications 13, 3133 (2022) <https://doi.org/10.1038/s41467-022-30719-4>`_ — one experimental PSF per channel, the channels registered to a reference channel, and all channels fitted jointly with linked parameters. Please cite that work when using multichannel spline fitting.

.. _localize-multichannel-spline-loading:

Loading the channels
~~~~~~~~~~~~~~~~~~~~

To build the calibration in the GUI, first load the channels:

- **Separate movies** — ``File`` > ``Open channels from several movies`` (or ``Open one multichannel movie`` for a single file holding several channels). The first movie loaded is the **reference channel**.
- **Split field of view** — if the channels are imaged side by side on one camera, load the single movie, tick **Regions = channels** in the ``Parameters`` dialog and drag the ROIs onto the channels. The first region is the reference channel. All regions are kept the same size: drag once to set the size, click to drop more, drag a region or use the arrow keys to fine-tune it.

In split field of view mode, each region also carries its **own identification settings**:

- Select a region (click it in the image, or its row in ``Edit ROIs...``) and the ``Min. net gradient`` slider, or the wavelet settings, show and tune *that* region alone — turn on ``Preview`` and sweep them as usual.
- With no region selected, they still set every region at once.
- The current threshold is drawn next to each region's ``ref`` / ``ch1`` label and listed in the ``min_ng`` (or ``wavelet thr.``) column of the ``Edit ROIs...`` table, where it can also be typed in directly.
- The per-region values are used for identification, for ``Calibrate spline PSF`` and for ``Re-align channels (current signal)``.

.. _localize-multichannel-spline-calibrating:

Calibrating
~~~~~~~~~~~

Run ``Calibration`` > ``Calibrate spline PSF`` as for a single channel (see :ref:`localize-spline-building`). The dialog is the same, with an additional option:

**Link photon counts across channels**
   On by default, so all channels share one photon count and background. Turn it off (2 to 6 channels) to fit per-channel photons and background instead, with only ``x``, ``y`` and ``z`` shared.

If photon counts are not linked, the resulting localizations contain per-channel columns, one set per channel ``c``:

- ``photons_ch<c>`` and ``bg_ch<c>`` — that channel's photon count and background. ``photons`` and ``bg`` are their sums.
- ``rel_photons_ch<c>`` — that channel's share of the total photons, so the values sum to 1 per localization.

Picasso builds a PSF for every channel and registers each non-reference channel to the reference by a transform estimated from matching beads; the per-channel PSFs and transforms are stored in one calibration ``.hdf5``.

- ``Channel registration`` in the calibration dialog chooses the model — ``translation`` (a pure xy shift), ``affine`` (the default), ``projective``, ``polynomial2`` or ``polynomial3`` — with the same trade-offs as the lateral corrections (see :ref:`localize-lateral-transform-models`).
- The choice is recorded in the calibration and used automatically at fit time, where each spot is linearized about its own position.
- Alongside the usual diagnostic plot, a ``<base>_registration.png`` is written showing how well the channels align (residuals and the decomposed shift / rotation / scale / mirror) — check it before fitting.

The registration is only as good as the bead stack it comes from:

- image **sparse beads**, so that a bead can only be paired with its own image in the other channel;
- image **several fields of view**, so that the correspondences cover the whole sensor;
- acquire the stack **on the same day as the measurement**, ideally directly before or after it to minimize the effect of drift. Registration from SMLM signal (not beads) is not straight-forward and difficult to control.

.. _localize-multichannel-spline-fitting:

Fitting
~~~~~~~

To fit, load the same channels, load the multichannel calibration under **Experimental PSF (spline)**, and run the fit with the ``Experimental PSF (cubic spline)`` model. Only spots detected in *every* channel are fitted, so identify each channel first, or identify once on the sum of the channels, which helps when brightness differs between channels (see :ref:`localize-identify-on-sum` below).

.. _localize-realign-channels:

Re-aligning the channels
~~~~~~~~~~~~~~~~~~~~~~~~

Multichannel spline fitting can benefit from re-aligning the channels on the data that is actually being fitted: the joint fit assumes each molecule maps onto the same ``x``, ``y``, ``z`` in every channel, so even a sub-pixel error in the transforms degrades the fit. If the calibration was taken right before or after the measurement, re-alignment is not required.

After identifying the channels, run ``Calibration`` > ``Re-align channels (current signal)`` to re-estimate the transforms from the blinking data itself. **This is strongly recommended whenever the bead stack and the measurement were not acquired directly one after another** (e.g. calibration from a previous day or session).

- Because the correction is derived by pairing the shared single-molecule signal frame by frame, a dialog first asks for the frame window to use and how many frames are evenly sampled from it.
- The result is reported per channel as the number of paired signals and the residual RMS (in camera pixels).
- We recommend using bright spots for the re-alignment: select a higher min. net gradient, or a higher wavelet threshold.

.. important::

   **The re-alignment updates the loaded calibration only; the calibration file is never modified.**

.. _localize-identify-on-sum:

Identifying on the sum of the channels
--------------------------------------

When one channel carries dimmer signal, identifying the channels separately may lose many molecules: the joint fit keeps only the spots found in *every* channel. The ``Identify on`` setting in the ``Parameters`` dialog (shown for multichannel and split-FOV data) offers two more modes for exactly this case:

**Each channel separately**
   The default described above.

**Sum of registered channels**
   Every channel is mapped onto the reference channel (see :ref:`localize-registering-channels`) and added up **in photons**, and the spots are identified in that sum. A molecule that is too faint to detect in any single channel can still stand out in the combined signal.

**Sum of unregistered channels**
   The channels are added up in photons pixel for pixel, as they are, without any registration (alignment). See :ref:`localize-summing-without-registration` below.

The sum modes also works for the multichannel 2D Gaussian fit, from a loaded channel registration exactly as it does from a spline calibration (see :ref:`localize-multichannel-gaussian`).

.. _localize-sum-registration:

Registration for the sum
~~~~~~~~~~~~~~~~~~~~~~~~

This section describes the identification on the registered sum, for which the channels have to be (re)registered *before* they can be summed (to add them up as they are, see :ref:`localize-summing-without-registration` below):

- **With a calibration loaded**, the registration comes from the loaded multichannel / split-FOV spline calibration — the sum is then built with exactly the transforms the fit will use.
- **Without a calibration**, Picasso first identifies every channel (or region) as usual and estimates the transforms from those detections, and only then builds the sum. This needs enough detections in every channel, so lower the minimum net gradient of the dim channel until spots appear in it.

A channel that cannot be registered is reported rather than assumed to be aligned, since summing it in at the wrong place would smear the very spots the mode is meant to recover.

.. _localize-sum-notes:

Things to be aware of
~~~~~~~~~~~~~~~~~~~~~

- **The sum has its own identification settings.** Box size, minimum net gradient and the identification filters are one set for the sum, shown in the ``Parameters`` dialog whenever a sum mode is selected, whichever channel is displayed.

  - Every channel keeps its own settings underneath: they come back when ``Identify on`` is set to ``Each channel separately``, and they are the ones the channels are identified with to register them for the sum.
  - The first time a sum is selected, it starts from the settings then shown. Both sum modes share the one set.

- **The threshold has to be re-tuned.** The sum is in photons and over all channels, so its net gradients are on a different scale than a single channel's raw counts. The wavelet threshold is relative to the noise of the sum, so it depends less on that scale, but the sum's noise and spots differ from a single channel's, so check it with ``Preview`` as well. In split-FOV mode the sum's single threshold (or set of wavelet settings) applies, as there is one summed image, and the per-region ones are left as they are.
- **The image on screen is the sum.** The display and ``Preview`` run on the summed movie, exactly as the identification does, so the threshold can be swept on what is actually being searched.

  - The summed view appears as soon as ``Sum of registered channels`` is selected, wherever the channels can be registered without identifying them first (a loaded calibration, or per-channel identifications already made) — so the minimum net gradient (or the wavelet settings) can be tuned with ``Preview`` before running ``Identify``.
  - If neither is available, the status bar says so and the summed view appears once ``Identify`` has identified the channels to register them.
  - For split-FOV data only the reference region is filled — the other regions have been mapped into it.

- **The fit does not link across channels.** The sum detections are already the cross-channel consensus, so they go into the joint fit as they are; requiring a detection in every channel on top of that would undo the whole point. The detections are in reference-channel coordinates, as the fit expects.
- The temporal median and Gaussian identification filters apply to the sum.
- The mode applies to the experimental data only. ``Calibrate spline PSF`` and the z-calibration are built from bead stacks one channel at a time and always identify the channels separately.

.. _localize-sum-realign:

Re-aligning before summing
~~~~~~~~~~~~~~~~~~~~~~~~~~

If the loaded calibration's registration is off, re-align it first: ``Calibration`` > ``Re-align channels (current signal)`` re-fits the inter-channel transform from the current blinking data (use a high ``Min. net gradient``, so only bright spots are paired) and updates the loaded calibration in memory.

- It keeps the calibration's own model unless another is picked in the dialog, and reports the model it actually fitted — if too few pairs survive for the model asked for, it falls back to an affine and says so.
- The channel sum is then built from the refined transforms — any sum made before the re-alignment is discarded, so run ``Identify`` again afterwards.

This is worth doing whenever the bead stack and the measurement were not acquired one after another, since the sum is only as sharp as the registration: a misaligned channel smears the summed spot and lowers its net gradient, which is exactly the signal the mode relies on.

.. _localize-summing-without-registration:

Summing without registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Sum of unregistered channels`` needs neither a calibration nor per-channel detections: the summed view is on screen as soon as the mode is selected, and switching between the channels shows the same image.

- Use it when the channels already overlay each other pixel for pixel (for example, the same field of view imaged one after another on one camera), or to look at the combined signal before any registration exists.
- Separate channel movies must then have the same frame size.
- If the channels do not overlay, each molecule shows up once per channel in the sum rather than as one brighter spot; use ``Sum of registered channels`` then.

The detections stand at the same pixel in every channel (or split-FOV region), which decides how they are fitted:

- ``Fit`` > **Each channel separately** fits every channel (or region) at these detections, each with its own settings, and saves one file per channel as usual (see :ref:`localize-analyzing-each-channel`).
- The joint fit takes them as it takes the registered sum's detections, without linking, but still places them in the other channels through the registration of the loaded calibration. This only makes sense when that registration is close to the identity, i.e. when the channels really do overlay.

.. tip::

   The same is available from a script via :func:`picasso.localize.identify_multichannel_sum` (and :class:`picasso.localize.SummedChannelsMovie` for the summed view itself); :func:`picasso.localize.unregistered_sum_transforms` gives the transforms that sum the channels without registration.
