.. _localize-phasor:

Phasor Localization
===================

Phasor localization (pSMLM, `Martens et al., J. Chem. Phys., 2018 <https://doi.org/10.1063/1.5005899>`_) finds the sub-pixel position of a spot without fitting a model to it. Each box is summed over its rows and over its columns, and the first Fourier coefficient of each of the two profiles is taken. The angle of the coefficient is the position of the spot along that axis, and its magnitude falls as the spot gets wider. The magnitudes of the x and y coefficients are the two **phasor magnitudes**. With a cylindrical lens, their ratio depends on z, so an astigmatic calibration gives z (pSMLM-3D).

The estimate is closed-form and non-iterative, which makes it very fast: millions of spots per second on a laptop CPU, so the identification takes almost all the time of a localization. It needs no PSF model and no starting values. A uniform background adds nothing to the first Fourier coefficient, so the position needs no background term either.

.. _localize-phasor-when:

When to use it
--------------

Use it when speed matters more than the last few percent of precision, for example for quick previews or very large data sets.

Consider the limits:

- **No widths and no goodness of fit.** There are no ``sx``, ``sy``, ``ellipticity``, ``log_likelihood`` or ``reduced_chi_square`` columns, and nothing to filter bad fits with besides the photons, the background and the phasor magnitudes.
- **Overlapping emitters** are not separated: a second spot in the box pulls the position toward it. For dense data, see :doc:`multi-emitter`.
- **The box matters.** It must contain the spot, but every extra pixel adds background noise. The paper's simulations (Sec. S3, S5) suggest 5 pixels for spots below about 1,000 photons, 7 above, and 11 for astigmatic 3D (at a pixel size of about 100 nm). A box much smaller than the spot cuts off its tails and biases the position toward the box center.
- It runs on the CPU only, and it fits one channel at a time. Channels and split-FOV regions can be fitted ``Separately``, each with its own settings, but not jointly.

.. _localize-phasor-gui:

In Picasso: Localize
--------------------

In the ``Parameters`` dialog, under ``Fit Settings``, select the model ``Phasor (pSMLM)``. Identify and fit as for any other model (see :ref:`localize-identification`).

.. _localize-phasor-output:

Output
------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Column
     - Content
   * - ``x``, ``y`` (nm)
     - The position from the angles of the first Fourier coefficients.
   * - ``photons``, ``bg``
     - Aperture photometry, as in the authors' implementation (paper, Sec. S11). The pixels within ``box // 2`` pixels of the box center are the signal, and the ring out to ``box // 2 + 2`` pixels is the background. ``bg`` is the 56th percentile of the ring (photons per pixel), and ``photons`` is the signal sum minus ``bg`` per signal pixel, at least 0. For small boxes, the ring holds the tails of the spot, so ``bg`` is high and ``photons`` is low.
   * - ``lpx``, ``lpy`` (nm)
     - The localization precision as **Picasso estimates it**; it is not part of the published method. See :ref:`localize-phasor-precision`.
   * - ``phasor_magnitude_x``, ``phasor_magnitude_y``
     - The magnitudes of the first Fourier coefficients of the x and y profiles, divided by the sum of the box (dimensionless, between 0 and 1). They fall as the spot widens along that axis, and with the background, which adds to the sum. Their ratio does not depend on the background or the brightness of the spot; it is what the 3D calibration uses.
   * - ``net_gradient``
     - As for the other models, for spots identified by their net gradient.
   * - ``z``, ``lpz`` (nm)
     - With a phasor calibration only, see :ref:`localize-phasor-3d`.

The ``Fit 2D`` entry of the metadata records ``Fit method: phasor`` and the box size. Its ``Localization precision`` and ``Photometry`` entries describe how those columns were obtained.

.. _localize-phasor-precision:

Localization precision
----------------------

.. important::

   The published phasor method gives **no localization precision**. Picasso needs one: every localization file has ``lpx`` and ``lpy`` columns, which Render, linking, G5M and other tools use. So Picasso **estimates** the precision itself.

The phasor position is a closed-form function of the pixel values, so the noise of each pixel can be propagated through it to first order ("delta method"). The variance of a pixel is its photon count, doubled for an EMCCD (its multiplication noise). With an sCMOS camera calibration loaded (see :ref:`localize-scmos-calibration`), the pixel's readout variance is added. With ``θ`` the angle of the first Fourier coefficient ``F`` of the x profile, ``k`` the pixel columns counted from the box center, ``ω = 2π / box`` and ``V[k]`` the summed variance of column ``k``:

.. math::

   \mathrm{var}(x) = \frac{\sum_k \sin^2(\omega k + \theta)\, V[k]}{|F|^2\, \omega^2}

and likewise for ``y``. This is the precision of the phasor estimator itself. It is **not a Cramer-Rao bound** (the best precision any estimator could reach) and needs no PSF model.

To compare with the precision measured from the data itself, check the NeNA value (see :ref:`render-nena`).

.. _localize-phasor-3d:

3D (astigmatism)
----------------

The phasor fit cannot use the Gaussian astigmatism calibration (see :doc:`3d-calibration`), which describes fitted widths. It has a calibration of its own, of the phasor magnitudes, following the paper (Sec. S6):

1. Record a bead z-stack as for the Gaussian calibration (see :ref:`localize-3d-calibration-gui`), select the ``Phasor (pSMLM)`` model and the box to use for the measurements, and select ``Calibration`` > ``Calibrate astigmatism (phasor)``. The same dialog asks for the step size, frames per step, frame order and z binning. The calibration is saved as ``<movie>_phasor_3d_calib.yaml`` by default, with its diagnostic plot next to it.
2. For each z step, the mean of each magnitude is taken, without the beads more than one standard deviation from that mean, as in the Gaussian calibration. The paper's quadratic, ``a (z - c)² + b`` (Eq. S5), is then fitted to each magnitude against the stage position. z = 0 is where the two curves cross. The **calibrated range** is the stretch of the scan around z = 0 over which the ratio of the two curves is monotonic, so that each ratio has only one z.
3. To fit z, load the calibration with ``Load calibration`` in the ``3D via Astigmatism`` group, as for the Gaussian one. ``Fit Z`` is then ticked, and the ``Magnification factor`` scales z (see :doc:`3d-calibration`).

For each localization, z is where the ratio of the two calibration curves equals the ratio of its magnitudes (Eq. S6). This is a quadratic equation, and the solution within the calibrated range is taken. A localization whose ratio has no solution in the range is removed, and the number removed is recorded in the metadata (``Outside calibrated z range``) and shown in the status bar. ``lpz`` is the propagated standard error of the log of the magnitude ratio, divided by the slope of the calibration's log ratio at z (and scaled by the magnification factor). The localizations of a calibrated fit are recorded as the ``Fit 3D phasor`` step, together with the calibration's coefficients, its range and the magnification factor.

The magnitudes depend on the box and on the PSF in camera pixels. A phasor calibration therefore records its ``Fit box size`` and ``Pixelsize``, and can only be used with both unchanged. The two calibration kinds are not interchangeable: a phasor calibration is refused by the Gaussian models, and the other way round.

.. _localize-phasor-cli:

Command line
------------

.. code-block:: console

   picasso localize movie.tif -a phasor -b 7 -px 130 -bl 100 -s 0.5
   picasso localize movie.tif -a phasor-3d -b 11 -zc movie_phasor_3d_calib.yaml -mf 0.79 -px 130 -bl 100 -s 0.5

``phasor-3d`` takes z from a phasor calibration (``-zc``); a Gaussian calibration is refused. A phasor calibration is made in Picasso: Localize or with ``zfit.calibrate_z_phasor``. ``phasor-3d`` cannot be combined with ``--regions-separately`` yet. See :ref:`cmd-localize`.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``fitting_method="phasor"`` selects the phasor fit in
   ``localize.localize`` (and ``localize.fit``):

   .. code-block:: python

      from picasso import io, localize

      movie, info = io.load_movie("movie.tif")
      camera_info = {
          "Baseline": 100, "Sensitivity": 0.5, "Gain": 1, "Pixelsize": 130
      }
      locs, info = localize.localize(
          movie,
          camera_info=camera_info,
          identification_parameters=localize.IdentificationParameters(box=7),
          movie_info=info,
          fitting_method="phasor",
      )
      io.save_locs("movie_locs.hdf5", locs, info)

   Calibrate from phasor localizations of a bead z-stack (10 nm steps),
   then fit z with the same box; ``calibration_3d`` takes the
   calibration or its path:

   .. code-block:: python

      from picasso import zfit

      calibration = zfit.calibrate_z_phasor(
          bead_locs, bead_info, d=10, magnification_factor=0.79,
          path="beads_phasor_3d_calib.yaml",
      )
      locs, info = localize.localize(
          movie,
          camera_info=camera_info,
          identification_parameters=localize.IdentificationParameters(box=11),
          movie_info=info,
          fitting_method="phasor",
          calibration_3d="beads_phasor_3d_calib.yaml",
      )

   The precision of spots cut out by hand, by the same propagation:

   .. code-block:: python

      from picasso.fitting import precision

      # spots: (n, box, box) photons; lpx, lpy in camera pixels
      lpx, lpy, sd_log_ratio = precision.phasor_precision(spots, em=False)

References
----------

- Martens, K. J. A., Bader, A. N., Baas, S., Rieger, B. & Hohlbein, J. Phasor based single-molecule localization microscopy in 3D (pSMLM-3D): An algorithm for MHz localization rates using standard CPUs. *J. Chem. Phys.* 148, 123311 (2018). https://doi.org/10.1063/1.5005899
