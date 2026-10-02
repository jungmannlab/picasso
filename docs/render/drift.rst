.. _render-drift-correction:

Drift Correction
================

Picasso offers three procedures to correct for drift:

- **Option A - AIM** (Ma, H., et al. Science Advances. 2024.). AIM is precise,
  robust, quick, requires no user interaction or fiducial markers (although
  adding them may improve performance).
- **Option B - specific structures in the image as drift markers.** It depends
  on the presence of either fiducial markers or inherently clustered
  structures in the image. On the other hand, it often supports more precise
  drift estimation than RCC and thus allows for higher image resolution.
- **Option C - an RCC algorithm.** It does not require any additional sample
  preparation.

To achieve the highest possible resolution (ultra-resolution), we recommend
AIM or consecutive applications of option C and multiple rounds of option B.

The drift markers for option B can be features of the image itself (e.g.,
protein complexes or DNA origami) or intentionally included markers (e.g., DNA
origami or gold nanoparticles). When using DNA origami as drift markers, the
correction is typically applied in two rounds:

1. first, with whole DNA origami structures as markers, and,
2. second, using single DNA-PAINT binding sites as markers.

In both cases, the precision of drift correction strongly depends on the
number of selected drift markers.

.. _render-aim:

Adaptive Intersection Maximization (AIM) drift correction
---------------------------------------------------------

1. In ``Picasso: Render``, select ``Postprocess > Undrift by AIM``.
2. The dialog asks the user to select:

   ``Segmentation``
      The number of frames per interval to calculate the drift. The lower the
      value, the better the temporal resolution of the drift correction, but
      the higher the computational cost.
   ``Intersection distance (nm)``
      The maximum distance between two localizations in two consecutive
      temporal segments to be considered the same molecule. This parameter is
      robust, 3*NeNA for optimal result is recommended.
   ``Max. drift in segment (nm)``
      The maximum expected drift between two consecutive temporal segments. If
      the drift is larger, the algorithm will likely diverge. Setting the
      parameter up to ``3 * intersection_distance`` will result in fast
      computation.

3. After the algorithm finishes, the estimated drift will be displayed in a
   pop-up window, and the display will show the drift-corrected image.

.. _render-marker-drift:

Marker-based drift correction
-----------------------------

1. In ``Picasso: Render``, pick drift markers as described in
   :ref:`render-picking`. Use the ``Pick similar`` option (see
   :ref:`render-pick-similar`) to automatically detect a large number of drift
   markers similar to a few manually selected ones.
2. If the structures used as drift markers have an intrinsic size larger than
   the precision of individual localizations (e.g., DNA origami, large protein
   complexes), it is critical to select a large number of structures.
   Otherwise, the statistic for calculating the drift in each frame (the mean
   displacement of localization to the structure's center of mass) is not
   valid.
3. Select ``Postprocess > Undrift from picked`` to compute and apply the drift
   correction. The command comes in two variants:

   - ``Undrift from picked (3D)`` performs drift correction using the picked
     localizations as fiducials. Also performs drift correction in z if the
     dataset has 3D information.
   - ``Undrift from picked (2D)`` performs drift correction using the picked
     localizations as fiducials. Does not perform drift correction in z even
     if dataset has 3D information.

4. (Optional) Save the drift-corrected localizations by selecting
   ``File > Save localizations``.

.. _render-rcc:

Redundant cross-correlation drift correction
--------------------------------------------

1. In ``Picasso: Render``, select ``Postprocess > Undrift by RCC``.
2. A dialog will appear asking for the segmentation parameter.

   - The default value, 1,000 frames, is a sensible choice for most movies.
   - It might be necessary to adjust the segmentation parameter of the
     algorithm, depending on the total number of frames in the movie and the
     number of localizations per frame.
   - A smaller segment size results in better temporal drift resolution but
     requires a movie with more localizations per frame.

3. After the algorithm finishes, the estimated drift will be displayed in a
   pop-up window, and the display will show the drift-corrected image.

.. _render-drift-tools:

Inspecting, undoing and reapplying drift
----------------------------------------

The ``Postprocess`` menu (see :doc:`menu-postprocess`) has three more drift
commands:

``Undo drift (2D)``
   Undo previous drift correction (only 2D part). Can be pressed again to
   redo.
``Show drift``
   After drift correction, a drift file is created. If the drift file is
   present, the drift can be displayed with this option.
``Apply drift from an external file``
   Applies drift from a user-specified .txt file.

.. note::

   The .txt drift files after consecutive undrifting rounds produce
   cumulative drift. Therefore, if 3 rounds of undrifting were performed, only
   the last file specifies the drift calculated in the 3 steps.
