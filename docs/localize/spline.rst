.. _localize-spline:

Experimental PSF (Cubic Spline)
===============================

Picasso can fit an **experimentally measured PSF** to every spot. The measured PSF is stored as a cubic spline — a smooth, piecewise-polynomial model built from a bead z-stack — and each spot is fit to that spline.

- This captures aberrations and engineered PSFs (e.g. astigmatism) that a Gaussian cannot describe.
- A 3D calibration recovers the axial position ``z`` directly: a single fit returns ``x``, ``y``, ``z``, photons and background, with no separate astigmatism z-calibration step.
- A 2D calibration models a single focal plane (no ``z``).

.. note::

   This feature is experimental — please report any unexpected behavior on our `GitHub issues page <https://github.com/jungmannlab/picasso/issues>`_.

Fitting runs on the CPU, or on any CUDA-capable GPU — see :doc:`gpu`; the kernels are compiled at run time by Numba, so no platform-specific binary is involved. *Building* a calibration follows the scheme of `Gpuspline <https://github.com/gpufit/Gpuspline>`_, its license is reproduced in ``LICENSES/Gpuspline-LICENSE.txt``.

**Localization precision.** The fit returns the fitted parameters but no uncertainties, so Picasso evaluates the Cramer-Rao lower bound separately to fill ``lpx``, ``lpy``, ``lpz``, ``photons_unc`` and ``bg_unc``. GPU with CUDA is used if detected, otherwise the process runs on the CPU.

The method combines three published works:

- the experimental-PSF localization workflow and bead alignment of `Li et al., Nature Methods 15, 367–369 (2018) <https://doi.org/10.1038/nmeth.4661>`_,
- the cubic-spline PSF model for single-molecule data introduced by `Babcock & Zhuang, Scientific Reports 7, 552 (2017) <https://doi.org/10.1038/s41598-017-00622-w>`_,
- the fitting algorithm of `Przybylski et al., Scientific Reports 7, 15722 (2017) <https://doi.org/10.1038/s41598-017-15313-9>`_.

The multichannel variant (see :ref:`localize-multichannel-spline` below) additionally follows the global-fitting approach of globLoc, `Li et al., Nature Communications 13, 3133 (2022) <https://doi.org/10.1038/s41467-022-30719-4>`_.

.. _localize-spline-building:

Building a spline calibration
-----------------------------

A calibration is built from a **bead z-stack**: image a sample of sparse, bright, sub-diffraction beads while scanning the stage through focus in even steps. Picasso then:

1. detects the beads (once, near focus — they are static in x/y),
2. cuts a box around each,
3. averages them across all beads and fields of view,
4. registers them in 3D,
5. normalizes the result to a clean PSF volume,
6. computes the cubic-spline coefficients.

This is the workflow described in `Li et al., Nature Methods 15, 367–369 (2018) <https://doi.org/10.1038/nmeth.4661>`_, however, PSF scaling was adapted to fit Gpufit's workflow.

.. tab-set::

   .. tab-item:: GUI

      Load the bead movie and select ``Calibration`` > ``Calibrate spline PSF``. A dialog collects:

      **Calibration step size (nm)**
         The axial stage step between consecutive frames (or z-positions).

      **Number of frames per step size** and **Frame order**
         For multi-FOV stacks that image several fields of view at each z-position (as in the 3D astigmatism dialog).

      **Z binning (steps per bin)** (default 1)
         Averages that many consecutive z-positions into one slice of the PSF model, so the spline's axial knots are *binning × step size* apart (the dialog shows the resulting bin size).

         - Each slice sits at the mean stage position of its steps; trailing steps that do not fill a whole slice are dropped.
         - This can mitigate spikes in the fitted ``z`` positions of single emitters at certain values, since the PSF model is smoother. For an astigmatic PSF, bins of about 50 nm remove these spikes without a measurable loss of precision, and the coarser model also fits faster. *Keep in mind that your system might work better with different binning!*
         - PSFs with fine axial structure (e.g. interference PSFs) need finer bins.
         - The diagnostic plot compares every single-frame bead spot with the stage position of its own frame, so binning does not inflate the reported precision.

      **Spline PSF model**
         ``3D (recovers z)`` or ``2D (single plane)``.

      **Magnification factor** (default 0.79)
         Scales the fitted ``z`` to correct for the refractive-index mismatch, as in the astigmatism fit (Huang et al., 2008). It is stored in the calibration and applied at fit time, not during calibration.

      **Set z = 0 at max. intensity**
         Define ``z = 0`` at the axial intensity peak of the averaged PSF instead of the center of the stage scan. Only meaningful for a PSF with a single, well-defined focus (e.g. astigmatism); off by default. This will impact the behavior of the magnification factor if the measured calibration data is offset.

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

   **The fit box size must not be larger than the box size the calibration was built with.** If they differ, Picasso Localize shows a dialog and offers to set the box size to the calibration's value (you then re-run identification before fitting).

.. _localize-spline-plot:

Reading the calibration plot
----------------------------

The diagnostic ``.png`` summarizes the averaged PSF and lets you judge the calibration at a glance. Its title reports the number of beads, the z range, the box and pixel size, and — when available — the model-vs-data agreement (median R² and NRMSE). Every image panel shares one intensity scale, and one camera pixel is drawn at the same physical size in all panels.

**xy slices (across z)**
   The PSF seen face-on at evenly spaced z-planes; the in-focus (sharpest) slice is outlined. A good calibration shows a compact, symmetric spot at focus that changes smoothly and symmetrically with defocus (for astigmatism, orthogonal elongation on either side of focus).

**xz and yz cross-sections**
   Side views with z on the vertical axis; a cyan line marks the sharpest slice. Look for a smooth, symmetric hourglass shape, without double-lobing or abrupt jumps between z-steps.

**Axial intensity profile** (always shown)
   The brightest normalized pixel per slice versus stage position. Expect a single clean peak, ≈ 1 at focus, decaying smoothly with defocus.

For a 3D calibration, Picasso also re-fits the individual beads through the new spline model (on the GPU when one is present, otherwise on the CPU) and adds three panels:

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

Each channel of a multichannel or split-FOV calibration builds its own PSF from its own beads and therefore filters independently: the inspector has a channel selector, and one gallery per channel is saved as ``<base>_ch{c}_beads.png``.

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
- for MLE, ``log_likelihood`` and ``iterations``.

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

In split field of view mode, each region also carries its **own minimum net gradient**:

- Select a region (click it in the image, or its row in ``Edit ROIs...``) and the ``Min. net gradient`` slider shows and tunes *that* region alone — turn on ``Preview`` and sweep it as usual.
- With no region selected, the slider still sets every region at once.
- The current value is drawn next to each region's ``ref`` / ``ch1`` label and listed in the ``min_ng`` column of the ``Edit ROIs...`` table, where it can also be typed in directly.
- The per-region values are used for identification, for ``Calibrate spline PSF`` and for ``Re-align channels (current signal)``.

.. _localize-multichannel-spline-calibrating:

Calibrating
~~~~~~~~~~~

Run ``Calibration`` > ``Calibrate spline PSF`` as for a single channel. The dialog is the same, with an additional option:

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
- image **several fields of view**, so that the correspondences cover the whole sensor rather than one part of it;
- acquire the stack **on the same day as the measurement**, ideally directly before or after it to minimize the effect of drift.

.. _localize-multichannel-spline-fitting:

Fitting
~~~~~~~

To fit, load the same channels, load the multichannel calibration under **Experimental PSF (spline)**, and run the fit with the ``Experimental PSF (cubic spline)`` model. Only spots detected in *every* channel are fitted, so identify each channel first.

.. _localize-realign-channels:

Re-aligning the channels
~~~~~~~~~~~~~~~~~~~~~~~~

Multichannel spline fitting benefits greatly from re-aligning the channels on the data that is actually being fitted: the joint fit assumes each molecule maps onto the same ``x``, ``y``, ``z`` in every channel, so even a sub-pixel error in the transforms degrades the fit.

After identifying the channels, run ``Calibration`` > ``Re-align channels (current signal)`` to re-estimate the transforms from the blinking data itself. **This is strongly recommended whenever the bead stack and the measurement were not acquired directly one after another** (e.g. calibration from a previous day or session).

- Because the correction is derived by pairing the shared single-molecule signal frame by frame, a dialog first asks for the frame window to use and how many frames are evenly sampled from it.
- The result is reported per channel as the number of paired signals and the residual RMS (in camera pixels).
- We recommend using bright spots for the re-alignment (simply select a higher min. net gradient).

.. important::

   **The re-alignment updates the loaded calibration only; the calibration file is never modified.**

.. _localize-identify-on-sum:

Identifying on the sum of the channels
--------------------------------------

When one channel carries very little signal, identifying the channels separately loses most molecules: the dim channel detects only a few of them, and the joint fit keeps only the spots found in *every* channel. The ``Identify on`` setting in the ``Parameters`` dialog (shown for multichannel and split-FOV data) offers two more modes for exactly this case:

**Each channel separately**
   The default described above.

**Sum of registered channels**
   Every channel is mapped onto the reference channel and added up in photons, and the spots are identified in that sum. A molecule that is too faint to detect in any single channel can still stand out in the combined signal.

**Sum of unregistered channels**
   The channels are added up in photons pixel for pixel, as they are, without any registration (alignment). See :ref:`localize-summing-without-registration` below.

The sum mode also works for the multichannel 2D Gaussian fit, from a loaded channel registration exactly as it does from a spline calibration (see :ref:`localize-multichannel-gaussian`).

.. _localize-sum-registration:

Registration for the sum
~~~~~~~~~~~~~~~~~~~~~~~~

The channels have to be (re)registered *before* they can be summed:

- **With a calibration loaded**, the registration comes from the loaded multichannel / split-FOV spline calibration — the sum is then built with exactly the transforms the fit will use.
- **Without a calibration**, Picasso first identifies every channel (or region) as usual and estimates the transforms from those detections, and only then builds the sum. This needs enough detections in every channel, so lower the minimum net gradient of the dim channel until spots appear in it.

A channel that cannot be registered is reported rather than assumed to be aligned, since summing it in at the wrong place would smear the very spots the mode is meant to recover.

.. _localize-sum-notes:

Things to be aware of
~~~~~~~~~~~~~~~~~~~~~

- **The sum has its own identification settings.** Box size, minimum net gradient and the identification filters are one set for the sum, shown in the ``Parameters`` dialog whenever a sum mode is selected, whichever channel is displayed.

  - Every channel keeps its own settings underneath: they come back when ``Identify on`` is set to ``Each channel separately``, and they are the ones the channels are identified with to register them for the sum.
  - The first time a sum is selected, it starts from the settings then shown. Both sum modes share the one set.

- **The minimum net gradient has to be re-tuned.** The sum is in photons and over all channels, so its net gradients are on a different scale than a single channel's raw counts. In split-FOV mode the sum's single threshold applies (there is one summed image), and the per-region thresholds are left as they are.
- **The image on screen is the sum.** The display and ``Preview`` run on the summed movie, exactly as the identification does, so the threshold can be swept on what is actually being searched.

  - The summed view appears as soon as ``Sum of registered channels`` is selected, wherever the channels can be registered without identifying them first (a loaded calibration, or per-channel identifications already made) — so the minimum net gradient can be tuned with ``Preview`` before running ``Identify``.
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
