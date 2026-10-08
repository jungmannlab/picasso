.. _localize-lateral-corrections:

Lateral Corrections (x, y)
==========================

Two things distort the lateral coordinates of a measurement:

- a cylindrical lens inserted for astigmatic 3D imaging may shift, rotate and stretch the image relative to the unmodified light path;
- chromatic aberration displaces one color channel relative to another.

Both are corrected the same way, by a geometric transform fitted from two bead images and applied to ``x`` / ``y`` after fitting.

.. _localize-lateral-calibrating:

Calibrating a lateral transform
-------------------------------

Open ``Calibration`` > ``Calibrate lateral transform (astigmatism / chromatic)...`` and choose what to correct:

- **Astigmatism (cylindrical lens)** — a reference image of in-focus beads *without* the cylindrical lens, and an image of the same beads *with* it.
- **Chromatic aberration** — an image of in-focus beads in the reference color channel, and an image of the same beads in the channel to be corrected.

The transform is then fitted as follows:

1. Beads are detected with the current ``Box side length`` and ``Min. net gradient`` (use ``Show`` to tune them on either image with a live preview).
2. The beads are refined by a 2D Gaussian fit and matched by mutual nearest neighbor.
3. A transform mapping the second image onto the reference is fitted by least squares.
4. Bead pairs whose residual is far from the median are dropped and the transform is refitted, so a single mismatched bead cannot warp the result.

.. _localize-lateral-transform-models:

Transform models
----------------

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
-------------------

A diagnostic figure is shown and saved next to the calibration as ``<base>_lateral_<type>.png``: overlays before and after the correction, and the mean per-bead cross-correlation before and after, whose peak should sit at the origin once the correction is applied.

After the fit, the bead pairing is drawn in the main window as color-coded identification boxes:

- Load either bead image (the ``Show`` buttons in the calibration dialog) and every detected bead is boxed.
- A bead and the bead it was matched with carry the **same color** in the reference and in the target image, while detections that stayed unmatched are gray.
- Hovering a box says which pair it belongs to.

This is the same reading as the cross-channel link colors used for multichannel data, and it makes a wrong or missing match visible on the data itself.

.. _localize-lateral-storage:

.. _localize-appending-or-loading-separately:

Where the transform is stored
-----------------------------

The transform is stored as one entry of an ordered ``Lateral transforms`` list in the calibration file you select. Both options give the **same coordinates**, so which one to use is a matter of bookkeeping:

- **Appended to a 3D or spline calibration** (recommended): an existing Gaussian 3D calibration (``.yaml``) or spline PSF calibration (``.hdf5``). The correction travels with the calibration it belongs to and is applied automatically whenever that calibration is used to fit, whether the fit is Gaussian astigmatism or cubic spline. There is nothing to load and nothing to forget.
- **A standalone lateral calibration** (``New``, a ``.yaml`` holding only lateral corrections), loaded separately at fit time (see :ref:`localize-lateral-applying` below). This is the only option for 2D data, where there is no 3D calibration to append to.

Corrections accumulate: calibrating both an astigmatism and a chromatic transform into the same file stores them as a list, and they are applied one after another in that order. Re-running a calibration of the same type replaces its entry rather than adding a second copy.

The two options can be combined. With astigmatic 3D fitting, the separately loaded corrections are applied after the z fit, on top of whatever the 3D calibration carries. So a 3D calibration holding the astigmatism correction plus a separately loaded chromatic one applies the astigmatism first and the chromatic second, exactly as if both had been appended to the same file.

.. warning::

   **The same correction is never applied twice.** Applying one twice would shift the coordinates twice, so a duplicate is dropped instead:

   - A file whose transform the loaded 3D or spline calibration already carries is refused at load time.
   - One that slips through as a copy saved under another name is skipped at fit time — the transforms themselves are compared, not the file names.
   - The same correction loaded more than once (the same file picked twice, or a copy of it) is applied once.

   Every correction that *was* applied is named in the saved metadata under ``Lateral corrections applied``.

.. important::

   **Lateral corrections apply to single-channel data only.** The multichannel (global) spline fit is a different mechanism: it fits all channels jointly and registers them itself from the per-channel transforms in its own calibration, so a lateral correction on top of that would be applied twice.

   Picasso therefore refuses to append a lateral transform to a multichannel spline calibration, and ignores loaded lateral corrections when a multichannel fit runs.

.. _localize-lateral-applying:

Loading a separate correction
-----------------------------

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
