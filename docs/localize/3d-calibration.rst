.. _localize-3d-calibration:

3D Calibration (Astigmatism)
============================

In astigmatic 3D imaging, a cylindrical lens makes the fitted spot widths ``sx`` and ``sy`` (i.e., the standard deviations :math:`\sigma` of the fitted Gaussians) depend on the axial position. Picasso calibrates this dependence from a bead z-stack and uses it to recover ``z`` from the spot widths. The cylindrical lens also shifts and distorts ``x`` and ``y``; see :doc:`lateral-correction` for how to correct this.

For better accuracy, use an experimentally measured PSF instead (see :doc:`spline`). A real astigmatic PSF is not an elliptical Gaussian and the spline model fits its actual shape and recovers ``z`` directly in the same fit as ``x`` and ``y``, rather than from the fitted widths afterwards.

.. _localize-3d-theory:

Theory
------

3D Calibration is performed by an adapted version of `Huang et al., 2008 <https://www.ncbi.nlm.nih.gov/pubmed/18174397/>`_.

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
