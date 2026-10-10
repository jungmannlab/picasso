.. _localize-3d-calibration:

3D Calibration (Astigmatism)
============================

In astigmatic 3D imaging, a cylindrical lens makes the fitted spot widths ``sx`` and ``sy`` (i.e., the standard deviations :math:`\sigma` of the fitted Gaussians) depend on the axial position. Picasso calibrates this dependence from a bead z-stack and uses it to recover ``z`` from the spot widths. The cylindrical lens also shifts and distorts ``x`` and ``y``; see :doc:`lateral-correction` for how to correct this.

For better accuracy, use an experimentally measured PSF instead (see :doc:`spline`). A real astigmatic PSF is not an elliptical Gaussian and the spline model fits its actual shape and recovers ``z`` directly in the same fit as ``x`` and ``y``, rather than from the fitted widths afterwards.

.. _localize-3d-calibration-gui:

Calibrating in the GUI
----------------------

1. Record a z-stack of fluorescent beads: move the stage through the focus in steps of known size (e.g., 10 nm).
2. Open the stack in ``Picasso: Localize`` and set ``Box side length`` and ``Min. net gradient`` in ``Analyze`` > ``Parameters...`` so that the beads are identified over the whole stack (check with ``Preview``). The temporal median filter is not applied during calibration, see :ref:`localize-temporal-median-filter`.
3. Select ``Calibration`` > ``Calibrate astigmatism (Gaussian)``. A dialog collects:

   **Calibration step size (nm)**
      The axial stage step between consecutive z positions.

   **Number of frames per step size** and **Frame order**
      For movies that image several fields of view (FOVs) to collect more beads: the number of FOVs, and whether all FOVs are imaged at each z position before the stage moves (``Different FOVs first``) or each FOV gets its own full z-stack (``Different z positions first``).

   **Z binning (steps per bin)** (default 1)
      See :ref:`localize-calibrating-z` below.

4. Choose where to save the calibration (``<movie>_3d_calib.yaml`` by default). The diagnostic plot is saved next to it, see :ref:`localize-3d-calibration-plot`.

To fit z, load the calibration with ``Load calibration`` in the ``3D via Astigmatism`` group of the ``Parameters`` dialog; ``Fit Z`` is then ticked. The ``Magnification factor`` (default 0.79) scales the fitted ``z`` to correct for the refractive-index mismatch between the immersion medium and the sample (`Huang et al., Science, 2008 <https://doi.org/10.1126/science.1153529>`__).

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   The calibration is fitted from the 2D-fitted bead localizations.
   ``calibrate_z`` saves the ``.yaml`` and the check plot (and opens the
   plot). ``zfit.zfit`` then adds ``z``, ``lpz`` and ``d_zcalib`` to a
   measurement; ``filter=2`` (the default) discards fits far from the
   calibration curves, ``filter=0`` keeps all. ``localize.localize`` fits z
   right after the 2D fit when given ``calibration_3d=``.

   .. code-block:: python

      from picasso import io, localize, zfit

      # Calibration: fit the bead z-stack in 2D first
      movie, info = io.load_movie("beads_zstack.tif")
      camera_info = {
          "Baseline": 100, "Sensitivity": 0.53, "Gain": 1, "Qe": 1, "Pixelsize": 130
      }
      locs, info = localize.localize(
          movie,
          camera_info=camera_info,
          identification_parameters={"Box Size": 7, "Min. Net Gradient": 5000},
          movie_info=info,
          fitting_method="gausslq",
      )
      calibration = zfit.calibrate_z(
          locs, info, d=10, magnification_factor=0.79, path="beads_3d_calib.yaml"
      )  # d: step size in nm

      # Fitting z for a measurement
      locs, info = io.load_locs("movie_locs.hdf5")
      calibration = io.load_calibration("beads_3d_calib.yaml")
      locs, info = zfit.zfit(locs, info, calibration=calibration)
      io.save_locs("movie_locs_3d.hdf5", locs, info)


.. _localize-3d-theory:

Theory
------

3D Calibration is performed by an adapted version of `Huang et al., 2008 <https://doi.org/10.1126/science.1153529>`_.

.. _localize-calibrating-z:

Calibrating z
-------------

After entering the step size, Picasso will calculate the mean and the variance of the spot widths :math:`s_x` and :math:`s_y` (``sx`` and ``sy``) for each z position. Localizations that are not within one standard deviation are discarded. A sixth-degree polynomial is fitted to the mean values of :math:`s_x` and :math:`s_y` (this deviates from the original publication slightly):

.. math::

   \bar{s}_x(z) &= c_x[6]\,z^0 + c_x[5]\,z^1 + \dots + c_x[0]\,z^6 \\
   \bar{s}_y(z) &= c_y[6]\,z^0 + c_y[5]\,z^1 + \dots + c_y[0]\,z^6

The calibration coefficients are stored in the YAML file and contain the coefficients :math:`c_x` and :math:`c_y`, the first entry being :math:`c[0]` and the last :math:`c[6]`.

**Z binning** (default 1) merges that many consecutive z positions into one axial bin before the polynomials are fitted:

- the mean widths of a bin's positions are averaged and placed at the mean of their stage positions;
- positions at the end of the scan that do not fill a whole bin are left out of the fit;
- the diagnostic plot still compares every localization with the stage position of its own frame.

Because the sixth-order polynomial is already smooth, binning changes an astigmatism calibration little; it matters mostly for the spline PSF (see :ref:`localize-spline-building`).

.. _localize-3d-calibration-plot:

Reading the calibration plot
----------------------------

When the calibration finishes, Picasso shows a six-panel diagnostic figure and saves it next to the calibration ``.yaml`` as a ``.png`` with the same base name.

The first three panels show how well the polynomial describes the beads; the last three show how well the resulting calibration recovers a known z. Spot widths and heights are in camera pixels, z and stage positions in nm.

**Mean spot width/height vs stage position**
   The measured mean ``sx`` and ``sy`` per z step with the two fitted sixth-degree polynomials on top. Picasso shifts the stage axis such that the two polynomial fits meet at ``z = 0``.

**Spot width vs spot height**
   Every kept localization (i.e., each bead at each z position) as a scatter, with the calibration curve through it. The cloud should follow the curve as a narrow band. A wide cloud means the beads disagree with each other (for example, field-dependent PSF or a tilted stage), and points far off the curve will be assigned a wrong z at fit time.

**Spot width/height vs estimated z**
   Similar to the first plot, however, each bead at each z position is shown.

**Estimated z vs stage position**
   The recovered z against the known stage position, with the identity line. Points should sit on the diagonal over the whole intended z range. The range where they do is the usable depth of the calibration; beyond it the points flatten out or fold back. The vertical spread of the scatter in this plot is reflected further in "Mean z precision vs stage position", see below.

**Deviation to true position**
   Histogram of ``estimated z − stage position`` over all localizations. It should be centered on 0 and single-peaked.

**Mean z precision vs stage position**
   The RMS deviation per z step. Note that these values are impacted by a tilted stage or field-dependent PSF! In this case, there is the data-driven range of the *real* z positions. Thus this is not necessarily the actual measure of axial localization precision.

.. note::

   These panels are computed from the calibration beads themselves, so they report how self-consistent the calibration is and may not reflect how it performs on dim single molecules.

.. _localize-fitting-z:

Fitting z
---------

For each localization, the spot widths :math:`s_x` and :math:`s_y` are fitted. As in Huang et al., ``z`` is then found by minimizing the distance :math:`D` between the measured widths and the calibration curves:

.. math::

   D(z) = \left(\sqrt{s_x} - \sqrt{w_x(z)}\right)^2 + \left(\sqrt{s_y} - \sqrt{w_y(z)}\right)^2

where :math:`w_x(z)` and :math:`w_y(z)` are the calibration polynomials :math:`\bar{s}_x(z)` and :math:`\bar{s}_y(z)` from above.
