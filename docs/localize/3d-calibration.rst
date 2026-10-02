.. _localize-3d-calibration:

3D Calibration (Astigmatism)
============================

In astigmatic 3D imaging, a cylindrical lens makes the fitted spot widths ``sx`` and ``sy`` depend on the axial position. Picasso calibrates this dependence from a bead z-stack and uses it to recover ``z`` from the spot widths. This page also covers the lateral corrections of ``x`` and ``y`` for astigmatism and chromatic aberration.

For an experimentally measured PSF that recovers ``z`` directly in the fit, see :doc:`spline`.

.. _localize-3d-theory:

Theory
------

3D Calibration is performed by an adapted version of `Huang et al., 2008 <https://www.ncbi.nlm.nih.gov/pubmed/18174397/>`_.

.. _localize-calibrating-z:

Calibrating z
-------------

After entering the step size, Picasso will calculate the mean and the variance for sigma_x and sigma_y for each z position. Localizations that are not within one standard deviation are discarded. A six-degree polynomial is fitted to the mean values of x and y:

.. math::

   \mathrm{mean\_sx} &= c_x[6]\,z^0 + c_x[5]\,z^1 + \dots + c_x[0]\,z^6 \\
   \mathrm{mean\_sy} &= c_y[6]\,z^0 + c_y[5]\,z^1 + \dots + c_y[0]\,z^6

The calibration coefficients are stored in the YAML file and contain the parameters of cx and cy. The first entry being c[0], the last being c[6].

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
   The measured mean ``sx`` and ``sy`` per z step with the two fitted six-degree polynomials on top. Picasso shifts the stage axis such that the two polynomial fits meet at ``z = 0``.

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

   These panels are computed from the calibration beads themselves, so they report how self-consistent the calibration is — not how it performs on dim single molecules, which will likely be worse.

.. _localize-fitting-z:

Fitting z
---------

For each localization, sigma_x and sigma_y is determined. Similar to the Science paper, the following equation is used to minimize the distance D:

.. math::

   D = \left(s_x^{0.5} - w_x^{0.5}\right)^2 + \left(s_y^{0.5} - w_y^{0.5}\right)^2

with w being :math:`c[6]\,z^0 + c[5]\,z^1 + \dots + c[0]\,z^6`.

.. _localize-lateral-corrections:

Lateral corrections of x and y
------------------------------

Two things distort the lateral coordinates of a measurement:

- a cylindrical lens inserted for astigmatic 3D imaging shifts, rotates and stretches the image relative to the unmodified light path;
- chromatic aberration displaces one color channel relative to another.

Both are corrected the same way, by a geometric transform fitted from two bead images and applied to ``x`` / ``y`` after fitting.

.. _localize-lateral-calibrating:

Calibrating a lateral transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open ``3D`` > ``Calibrate lateral transform (astigmatism / chromatic)`` and choose what to correct:

- **Astigmatism (cylindrical lens)** — a reference image of in-focus beads *without* the cylindrical lens, and an image of the same beads *with* it.
- **Chromatic aberration** — an image of in-focus beads in the reference color channel, and an image of the same beads in the channel to be corrected.

The transform is then fitted as follows:

1. Beads are detected with the current ``Box side length`` and ``Min. net gradient`` (use ``Show`` to tune them on either image with a live preview).
2. The beads are refined by a 2D Gaussian fit and matched by mutual nearest neighbor.
3. A transform mapping the second image onto the reference is fitted by least squares.
4. Bead pairs whose residual is far from the median are dropped and the transform is refitted, so a single mismatched bead cannot warp the result.

.. _localize-lateral-transform-models:

Transform models
~~~~~~~~~~~~~~~~

``Transform model`` chooses how the two frames are related:

**Translation** (2 DOF, at least 1 bead pair)
   A shift in x and y and nothing else. The right choice when the two frames are known to differ only by an offset: with one free parameter per axis it is the least noise-prone of the models, and a rotation or scale it cannot absorb shows up in the residual instead of being fitted away.

**Affine** (6 DOF, at least 3 bead pairs)
   Translation, rotation, scale and shear. The default, and what a well-aligned optical path does to first order.

**Projective** (8 DOF, at least 4 pairs)
   Adds the perspective (keystone) term that a tilted dichroic or an unequal path length introduces. The residual an affine leaves grows towards the edges of the field, which is exactly what this removes.

**Polynomial2 / Polynomial3** (at least 6 / 10 pairs)
   A smooth warp of that degree that follows genuine field distortion.

   - This is not an optical model, and it extrapolates badly outside the region the beads span, so use it only with many, well-spread beads.
   - Its reverse map is fitted independently rather than inverted algebraically, so round-tripping a coordinate is accurate only to the round-trip RMS reported with the calibration; no fitted coordinate depends on that reverse map.

The stated minima are hard requirements — fitting fails below them — but about three times as many pairs are wanted, otherwise the transform interpolates the noise in the bead positions instead of averaging it out.

.. _localize-lateral-diagnostics:

Checking the result
~~~~~~~~~~~~~~~~~~~

A diagnostic figure is shown and saved next to the calibration as ``<base>_lateral_<type>.png``: overlays before and after the correction, and the mean per-bead cross-correlation before and after, whose peak should sit at the origin once the correction is applied.

After the fit, the bead pairing is drawn in the main window as color-coded identification boxes:

- Load either bead image (the ``Show`` buttons in the calibration dialog) and every detected bead is boxed.
- A bead and the bead it was matched with carry the **same color** in the reference and in the target image, while detections that stayed unmatched are gray.
- Hovering a box says which pair it belongs to.

This is the same reading as the cross-channel link colors used for multichannel data, and it makes a wrong or missing match visible on the data itself.

.. _localize-lateral-storage:

Where the transform is stored
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transform is stored as one entry of an ordered ``Lateral transforms`` list in the calibration file you select, which can be:

- an existing Gaussian 3D calibration (``.yaml``) or spline PSF calibration (``.hdf5``) — the transform is appended to it and applied automatically whenever that calibration is used to fit, whether the fit is Gaussian astigmatism or cubic spline;
- a standalone lateral calibration (``New``, a ``.yaml`` holding only lateral corrections) — loaded separately at fit time, and the only route for 2D data, where there is no 3D calibration to append to.

Corrections accumulate: calibrating both an astigmatism and a chromatic transform into the same file stores them as a list, and they are applied one after another in that order. Re-running a calibration of the same type replaces its entry rather than adding a second copy.

.. _localize-appending-or-loading-separately:

Appending or loading separately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both routes give the **same coordinates**, so which one to use is a matter of bookkeeping:

- **Appended to the 3D or spline calibration** (recommended) — the correction travels with the calibration it belongs to and is applied automatically whenever that calibration is used to fit. There is nothing to load and nothing to forget.
- **Loaded separately** — the standalone ``.yaml`` is loaded at fit time (see :ref:`localize-lateral-applying` below).

The two mix. With astigmatic 3D fitting, the separately loaded corrections are applied after the z fit, on top of whatever the 3D calibration carries. So a 3D calibration holding the astigmatism correction plus a separately loaded chromatic one applies the astigmatism first and the chromatic second, exactly as if both had been appended to the same file.

The same correction is never applied twice:

- A file whose transform the loaded 3D or spline calibration already carries is refused at load time.
- One that slips through as a copy saved under another name is skipped at fit time — the transforms themselves are compared, not the file names.
- Every correction that *was* applied is named in the saved metadata under ``Lateral corrections applied``.

.. important::

   **Lateral corrections apply to single-channel data only.** The multichannel (global) spline fit is a different mechanism: it fits all channels jointly and registers them itself from the per-channel transforms in its own calibration, so a lateral correction on top of that would be applied twice.

   Picasso therefore refuses to append a lateral transform to a multichannel spline calibration, and ignores loaded lateral corrections when a multichannel fit runs.

.. _localize-lateral-applying:

Loading a separate correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::

   .. tab-item:: GUI

      Load the standalone ``.yaml`` through the ``Lateral correction (x, y)`` box in the ``Parameters`` dialog:

      - ``Load correction`` takes one or more files (applied in the order listed);
      - ``Clear`` drops them.

      The setting belongs to the loaded movie, so several movies opened side by side can each carry their own correction.

   .. tab-item:: Command line

      ``picasso localize`` takes ``--affine-calibration <file>`` (repeat the flag to chain several); whichever model the file stores is used as saved. It combines with ``--zc`` the same way the GUI does:

      .. code-block:: bash

         picasso localize movie.tif -zc astig_3d_calib.yaml -ac chromatic.yaml

   .. tab-item:: Python

      ``localize.localize`` takes ``affine_calibration`` alongside ``calibration_3d``, and ``zfit.zfit`` takes ``lateral_transforms`` for localizations that are already fitted. Both accept a calibration dictionary, a list of entries or a path, and both skip (with a ``DuplicateLateralTransformWarning``) a correction the 3D calibration already carries:

      .. code-block:: python

         locs, info = zfit.zfit(
             locs,
             info,
             calibration=z_calibration,
             lateral_transforms="chromatic.yaml",
         )
