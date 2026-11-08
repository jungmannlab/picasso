.. _localize-multi-emitter:

Overlapping Emitters (Multi-Emitter Fitting)
============================================

Every other fit of Picasso Localize places a box around each identified spot and fits **one** emitter to it. Two emitters closer than about 2 PSF σ in the same frame (σ is the standard deviation of the PSF; its FWHM is 2.355 σ) then give either one localization between them or a poor fit (a high ``reduced_chi_square``). This happens more frequently in data with high density of blinking.

The **multi-emitter fit** localizes such emitters separately. Picasso implements the 3D-DAOSTORM analysis of `Babcock, et al. Optical Nanoscopy, 2012 <https://doi.org/10.1186/2192-2853-1-6>`_ with three of its PSF models (see :ref:`localize-multi-emitter-models`), ported from its reference implementation, `storm-analysis <https://github.com/ZhuangLab/storm-analysis>`_, whose license is reproduced in `LICENSES/storm-analysis-LICENSE.txt <https://github.com/jungmannlab/picasso/blob/master/LICENSES/storm-analysis-LICENSE.txt>`__, and its cubic spline version of `Babcock & Zhuang, Scientific Reports, 2017 <https://doi.org/10.1038/s41598-017-00622-w>`_, with an experimental PSF from a spline calibration. Picasso adds three things to the Gaussian models: the PSF is integrated over the pixels, as Simulate's movies are, instead of sampled at the pixel centers; the PSF σ can be estimated from the movie; and every emitter gets a localization precision.

It fits whole frames instead of boxes, with all their emitters at once:

1. The **background** of the frame is estimated by a broad Gaussian low pass of the frame minus the current fit.
2. **Peaks** are found in what the fit does not explain yet: the frame minus the background and the fit, smoothed with the PSF and divided by its expected noise, i.e., in units of the signal-to-noise ratio (SNR). A peak must exceed the ``SNR threshold`` and be the maximum within the ``Peak search radius``.
3. The new peaks join the fit. Each emitter is fitted by maximum likelihood, with every other emitter in place; emitters near a new or a removed one are fitted again until all have converged.
4. Of two emitters closer than 1 PSF σ, the dimmer one is removed, and so is every emitter whose significance does not exceed the ``SNR threshold``.

The cycles repeat until a cycle finds no new peak, or up to ``Max. cycles``. The PSF is either a 2D Gaussian integrated over the pixels, whose σ is fixed for the whole movie or fitted for each emitter, or a measured PSF.


.. _localize-multi-emitter-when:

When to use it
--------------

As per the 2D datasets of the SMLM Challenge 2016 (`Sage et al., Nature Methods, 2019 <https://doi.org/10.1038/s41592-019-0364-4>`_), use it for **dense 2D data with the emitters in focus**. It also should do no harm on sparse frames when the parameters are set right. For **3D data**, use it with the ``Fitted sx and sy`` model for an astigmatic PSF, or with a spline calibration of any PSF (see :ref:`localize-multi-emitter-models`).

Consider the limits:

- **Close pairs.** Of two emitters closer than 1 PSF σ only the brighter is kept, so pairs closer than about 1.5 PSF σ are seldom resolved. Whether a pair farther apart is resolved depends on its brightness and the ``SNR threshold``: in simulations with a fixed σ (1 camera pixel, background 30 photons), pairs of 5000-photon emitters 2 σ apart were all resolved at the default threshold of 15, but pairs of 1000-photon emitters only from 3 σ apart; at a threshold of 6, they were all resolved from 2 σ.
- **Emitters out of focus.** With a fixed σ, a spot wider than the PSF is fitted as several narrow emitters. On the SMLM Challenge's 1.5 µm thick sample, emitters more than 200 nm out of focus got about one false neighbor each. A fitted σ does not split them, but it finds fewer emitters (see :ref:`localize-multi-emitter-models`). For 3D data, use the ``Fitted sx and sy`` model or a 3D spline calibration.
- **Bright emitters and non-Gaussian PSFs.** The rings of a real PSF around a very bright emitter can be found as faint emitters next to it. Raise the ``SNR threshold``, or filter by ``significance`` afterwards (see :ref:`filter-expression`).
- **Structured background.** Out-of-focus fluorescence that changes from frame to frame is found as dim emitters wherever the background filter does not follow it. Tune the threshold with the preview, or filter by ``significance`` afterwards.
- It runs on the CPU, frame by frame on several threads, and fits one channel at a time (no multichannel or split-FOV fitting yet). Only the z fit of an astigmatic calibration runs on a CUDA GPU: in Picasso: Localize whenever one is available, on the command line with ``--fit-z-gpu``.

.. _localize-multi-emitter-models:

PSF models
----------

The ``PSF model`` (``psf_model`` in Python, ``--psf-model`` on the command line) sets the PSF of each emitter:

.. list-table::
   :header-rows: 1
   :widths: 22 40 14 24

   * - Model
     - PSF of each emitter
     - storm-analysis
     - Use for
   * - ``Fixed sigma`` (``fixed``, the default)
     - the ``PSF sigma`` of the movie, ``sx = sy``
     - ``2dfixed``
     - 2D data
   * - ``Fitted sigma`` (``spherical``)
     - one σ per emitter, ``sx = sy``
     - ``2d``
     - 2D data where the width of each localization is wanted
   * - ``Fitted sx and sy`` (``elliptical``)
     - a σ in x and one in y per emitter
     - ``3d``
     - astigmatic 3D data: z is read from ``sx`` and ``sy`` with a 3D calibration
   * - ``Experimental PSF (cubic spline)`` (``spline``)
     - the PSF of a spline calibration, at a fitted z for a 3D one
     - ``spliner``
     - any PSF, 2D or 3D; z is fitted directly

A fitted σ starts at the ``PSF sigma`` and stays within 0.5 to 5 times it, and each emitter is fitted in a region 1.5 times as large, as in storm-analysis.

**A fitted σ finds fewer emitters.** After the first emitter of a close pair is found, a fitted σ widens to cover both, so the residual shows no second peak and the pair ends up as one wide localization.

**Astigmatic 3D.** With ``Fitted sx and sy`` and a 3D calibration (see :ref:`localize-3d-calibration`), z is read from each emitter's ``sx`` and ``sy`` after the fit, as storm-analysis's ``3d`` model does.

**Spline PSF.** With ``Experimental PSF (cubic spline)``, every emitter is the PSF of a single-channel spline calibration (see :ref:`localize-spline`), sampled at its pixels, with its photons, x, y, background and, for a 3D calibration, z fitted, as in Babcock & Zhuang (2017). It is implemented from the paper, not ported from storm-analysis's ``spliner``. Peaks are found by filtering with the PSF at the ``Peak search z`` (by default only at 0 nm), and a new emitter starts at the z where it is brightest; the paper found more than one plane useful only for PSFs that change their shape with z beyond a blur, such as the double helix. z stays within the calibrated range. Each emitter is fitted in a disc of ``Fit region`` pixels, cropped from the calibration as the box of the single-emitter spline fit is, so a calibration with a large box, e.g., 25 pixels, can be fitted in a smaller region; the region must be at least 2 pixels smaller than the calibration box. ``Auto`` takes it from the calibration's Gaussian-equivalent σ in focus, as for the Gaussian models: 8 σ, or 12 σ for a 3D calibration, whose PSF widens with z (e.g., 12 pixels for σ = 1 pixel). The peaks are found with the PSF within the same region. The Gaussian-equivalent σ takes the place of the ``PSF sigma`` in removing close emitters and in ``n_emitters``.

.. _localize-multi-emitter-gui:

In Picasso: Localize
--------------------

In the ``Parameters`` dialog, under ``Fit Settings``, select the model ``Overlapping emitters (3D-DAOSTORM)``. The optimizer is MLE; the identification settings (box, method, filters) disappear, since the fit finds the spots itself, while the ROIs and the frame range still apply. Its settings are under ``Overlapping emitters``:

- ``PSF model`` (default ``Fixed sigma``): the PSF of each emitter, see :ref:`localize-multi-emitter-models`. With ``Fitted sx and sy``, the ``3D via Astigmatism`` section appears, to get z as for the single-emitter fits (see :ref:`localize-3d-calibration`). With ``Experimental PSF (cubic spline)``, the ``Experimental PSF (spline)`` section appears instead, to load the calibration (see :ref:`localize-spline-fitting`), and the ``PSF sigma`` gives way to the ``Fit region`` (``Auto`` by default) and, for a 3D calibration, the ``Peak search z (nm)``: the z of the filter planes, separated by commas.
- ``SNR threshold`` (default 15): the peak threshold and the minimum significance of an emitter. storm-analysis uses 6, which on real data also finds dim spots on out-of-focus background; on dense frames with a flat background, lower values (down to 6) find more emitters.
- ``PSF sigma (nm)`` (Gaussian models): σ for a fixed σ, and the start of a fitted one; peaks are always found with it. With ``Estimate`` ticked (the default), σ is estimated from the movie as the median σ of isolated emitters fitted with a free σ; as soon as σ is estimated, before the frames are fitted, it is shown in the spin box and in the status bar. Untick ``Estimate`` to give σ.
- ``Max. cycles`` (default 20): the most find-fit-subtract cycles per frame. A frame stops earlier, at the first cycle that finds no new peak; sparse frames need two or three cycles. A lower value saves time on very dense frames but leaves the emitters that later cycles would have found.
- ``Peak search radius`` (default 5 camera pixels): a new peak must be the highest point of the SNR map within this radius. A peak next to a brighter one is therefore not found in the same cycle, but in a later one, once the brighter emitter is fitted and subtracted. A smaller radius finds close neighbors in fewer cycles, but also more noise.
- ``Background filter sigma`` (default 8 camera pixels): the σ of the Gaussian low pass that estimates the background from the frame minus the current fit. It must be much wider than the PSF, so that an emitter does not count as its own background. A smaller value follows a background that changes over shorter distances, such as out-of-focus fluorescence in cells, but in dense regions it starts to take emitters' photons for background.

The three are lengths on the pixel grid the filters work on, hence in camera pixels, and rarely need changing.

Run ``Analyze`` > ``Localize (Identify & Fit)`` or ``Analyze`` > ``Fit``; there is nothing to identify first. With ``Preview`` ticked, the current frame is fitted with the settings shown, so the threshold can be tuned frame by frame. Emitters with a neighbor within 4 PSF σ are marked with a sky blue ``+``, the others with the usual green ``x``.

.. _localize-multi-emitter-output:

Output
------

One localization per emitter, with the columns of an MLE fit (see :ref:`files-localization-hdf5`) and two more:

- ``significance``: the signal of the emitter over the square root of the background it sits on, both weighted by the PSF.
- ``n_emitters``: the number of emitters within 4 PSF σ, the emitter itself included, counting at most its 8 nearest neighbors (so at most 9); 1 for an isolated emitter. These are the emitters its precision is computed from (see below). To keep the isolated ones only, filter by ``n_emitters == 1``.

For the Gaussian models, ``sx`` and ``sy`` hold the PSF σ, fixed or fitted; a fitted σ adds its uncertainties ``sx_unc`` and ``sy_unc``. With a 3D calibration, ``z``, ``d_zcalib`` and ``lpz`` are added, as for the single-emitter fits. The spline has no ``sx`` and ``sy``, and a 3D spline calibration adds ``z`` and ``lpz``, as the single-emitter spline fit does. The localization precisions ``lpx``, ``lpy`` and, for the spline, ``lpz``, and the uncertainties ``photons_unc``, ``bg_unc``, ``sx_unc`` and ``sy_unc`` are Cramer-Rao bounds, as in `Huang et al., Biomed. Opt. Express, 2011 <https://doi.org/10.1364/BOE.2.001377>`_: each comes from the joint Fisher matrix of the emitter and its neighbors within 4 PSF σ (at most the 8 nearest), so the bounds grow where emitters overlap; a fitted σ or z is among the parameters of each emitter. Where that matrix is singular, Huang et al.'s approximation (their Eq. 11) is used instead. ``log_likelihood`` and ``reduced_chi_square`` describe the fit of the region around the emitter, its neighbors included. The settings, and the PSF σ that was used (or, for a fitted σ, started from), are recorded in the ``Fit multi-emitter`` entry of the metadata, for the spline with the fit region that was used and the calibration's model and path.

.. _localize-multi-emitter-cli:

Command line
------------

.. code-block:: console

   picasso localize movie.tif --multi-emitter -px 130 -bl 100 -s 0.5
   picasso localize movie.tif --multi-emitter -px 130 -bl 100 -s 0.5 --psf-model elliptical -zc 3d_calib.yaml -mf 0.79
   picasso localize movie.tif --multi-emitter -px 130 -bl 100 -s 0.5 --psf-model spline -sc spline_calib.hdf5 --fit-region-size 11

Further options: ``--psf-model`` (``fixed``, ``spherical``, ``elliptical`` or ``spline``; with ``elliptical``, ``--z-calibration`` gives z, fitted on a CUDA GPU with ``--fit-z-gpu``; ``spline`` takes one ``--spline-calibration``), ``--peak-search-z`` (nm, for a 3D spline; 0), ``--fit-region-size`` (camera pixels, for the spline; from the calibration), ``--snr-threshold`` (15), ``--psf-sigma`` (nm; estimated by default), ``--max-cycles`` (20), ``--peak-radius`` (5), ``--background-sigma`` (8) and ``--seed-identification``, which starts each frame from the spots the identification options find instead of the first cycle's peaks. See :ref:`cmd-localize`.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``localize_multi_emitter`` takes the movie, its metadata and the camera
   information, and the settings as one
   ``localize.MultiEmitterParameters``; without them, the defaults apply
   and the PSF σ is estimated.

   .. code-block:: python

      from picasso import io, localize

      movie, info = io.load_movie("movie.tif")
      camera_info = {
          "Baseline": 100, "Sensitivity": 0.5, "Gain": 1, "Pixelsize": 130
      }
      locs, info = localize.localize_multi_emitter(
          movie,
          info,
          camera_info=camera_info,
          parameters=localize.MultiEmitterParameters(snr_threshold=15.0),
          progress="console",
      )
      io.save_locs("movie_locs.hdf5", locs, info)
      # the PSF σ the fit used, in nm
      print(info[-1]["PSF Sigma (nm)"])

   Astigmatic 3D: a σ in x and one in y per emitter, and z from a 3D
   calibration, or its path (``fit_z_gpu=True`` fits z on a CUDA GPU):

   .. code-block:: python

      locs, info = localize.localize_multi_emitter(
          movie,
          info,
          camera_info=camera_info,
          parameters=localize.MultiEmitterParameters(psf_model="elliptical"),
          calibration_3d="3d_calib.yaml",
      )

   A measured PSF: a spline calibration, or its path; a 3D one gives z.
   Here each emitter is fitted in a disc of 11 pixels:

   .. code-block:: python

      locs, info = localize.localize_multi_emitter(
          movie,
          info,
          camera_info=camera_info,
          parameters=localize.MultiEmitterParameters(
              psf_model="spline", fit_region_size=11
          ),
          spline_calibration="spline_calib.hdf5",
      )

References
----------

- Babcock, H., Sigal, Y. M. & Zhuang, X. A high-density 3D localization algorithm for stochastic optical reconstruction microscopy. *Optical Nanoscopy* 1, 6 (2012). https://doi.org/10.1186/2192-2853-1-6
- Babcock, H. P. & Zhuang, X. Analyzing single molecule localization microscopy data using cubic splines. *Scientific Reports* 7, 552 (2017). https://doi.org/10.1038/s41598-017-00622-w
- storm-analysis, Zhuang Lab, Harvard University, v2.4.1. https://github.com/ZhuangLab/storm-analysis
- Huang, F., Schwartz, S. L., Byars, J. M. & Lidke, K. A. Simultaneous multiple-emitter fitting for single molecule super-resolution imaging. *Biomed. Opt. Express* 2, 1377–1393 (2011). https://doi.org/10.1364/BOE.2.001377
- Sage, D. et al. Super-resolution fight club: assessment of 2D and 3D single-molecule localization microscopy software. *Nature Methods* 16, 387–395 (2019). https://doi.org/10.1038/s41592-019-0364-4
