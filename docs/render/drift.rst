.. _render-drift-correction:

Drift Correction
================

SMLM measurements often suffer from sample drift due to their relatively long acquisition times and the setup instability (e.g., thermal noise). Picasso offers three procedures to correct for drift:

- **Option A - AIM** (`Ma et al., Science Advances, 2024
  <https://doi.org/10.1126/sciadv.adm7765>`_). AIM is precise,
  robust, quick, requires no user interaction or fiducial markers (although
  adding them may improve performance).
- **Option B - specific structures in the image as drift markers.** Called "Undrift from picked" in the GUI. It depends
  on the presence of either fiducial markers or inherently clustered
  structures in the image. On the other hand, it often supports more precise
  drift estimation than RCC and thus allows for higher image resolution.
- **Option C - an RCC algorithm** (`Wang et al., Optics Express, 2014 <https://doi.org/10.1364/OE.22.015982>`_). It does not require any additional sample preparation.

To achieve the highest possible resolution, we recommend AIM and multiple rounds of option B (if available).

The drift markers for option B can be features of the image itself which constist of individual docking strands imaged (e.g., protein complexes or DNA origami) or intentionally included markers (e.g., DNA origami or gold nanoparticles). The precision of drift correction in this method depends on the number of selected drift markers.

.. _render-aim:

Adaptive Intersection Maximization (AIM) drift correction
---------------------------------------------------------

Please refer to `Ma et al., Science Advances, 2024 <https://doi.org/10.1126/sciadv.adm7765>`_ for detail on the mechanism of AIM.

1. In ``Picasso: Render``, select ``Postprocess > Undrift by AIM``.
2. The dialog asks the user to select:

   ``Segmentation``
      The number of frames per interval to calculate the drift. The lower the
      value, the better the temporal resolution of the drift correction, but
      the higher the computational cost. **Note: too low segmentation leads to
      no overlaps between the consecutive segments which causes the algorithm
      to fail.**
   ``Intersection distance (nm)``
      The maximum distance between two localizations in two consecutive
      temporal segments to be considered the same molecule. This parameter is
      robust, however, for optimal results, :math:`3 \times \mathrm{NeNA}` is recommended (see
      :ref:`NeNA <render-nena>`).
   ``Max. drift in segment (nm)``
      The maximum expected drift between two consecutive temporal segments. If
      the drift is larger, the algorithm will likely diverge. Setting the
      parameter up to ``3 * intersection_distance`` will result in fast
      computation.

3. After the algorithm finishes, the estimated drift will be displayed in a
   pop-up window, and the display will show the drift-corrected image.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   The distances are in camera pixels, so divide the dialog's nm values by
   the pixel size. ``roi_r`` is ``Max. drift in segment``.

   .. code-block:: python

      from picasso import aim, io, lib

      locs, info = io.load_locs("movie_locs.hdf5")
      pixelsize = lib.get_from_metadata(info, "Pixelsize")
      locs, info, drift = aim.aim(
          locs, info,
          segmentation=100,              # frames
          intersect_d=20 / pixelsize,
          roi_r=60 / pixelsize,
          progress="console",
      )
      io.save_locs("movie_locs_aim.hdf5", locs, info)
      io.save_drift("movie_locs_aim_drift.txt", drift)


.. _render-marker-drift:

Marker-based drift correction
-----------------------------

Static structures that are imaged throughout the movie, such as fiducial
markers or DNA origami, serve as drift markers. In every pick, each
localization's offset from the pick's center of mass is treated as its drift in that
frame. The drift of a frame is then the average over all picks with a
localization in it, each pick weighted by how closely it follows the common
drift, so that a noisy marker counts less. Frames without any localization in
the picks are interpolated linearly. For an example, see Fig. 4 of
`Schnitzbauer et al., Nature Protocols, 2017 <https://doi.org/10.1038/nprot.2017.024>`_.

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
   correction. If 3D data is detected, the user can choose whether or not to
   undrift in z.
4. (Optional) Save the drift-corrected localizations by selecting
   ``File > Save localizations``.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``undrift_from_picked`` estimates the drift from the picked
   localizations (one DataFrame per pick, see :doc:`picking`), and
   ``apply_drift`` subtracts it. ``undrift_from_fiducials`` finds the
   fiducials itself (``Tools > Pick fiducials``) and does both.

   .. code-block:: python

      from picasso import io, postprocess

      locs, info = io.load_locs("movie_locs_aim.hdf5")
      picked = postprocess.picked_locs(locs, info, picks, "Circle", pick_size=radius)
      drift = postprocess.undrift_from_picked(picked, info)    # columns x, y (and z)
      locs = postprocess.apply_drift(locs, info, drift=drift)

      # or, with automatically picked fiducials
      locs, info, drift = postprocess.undrift_from_fiducials(locs, info)


.. _render-rcc:

Redundant cross-correlation (RCC) drift correction
--------------------------------------------------

Please refer to `Wang et al., Optics Express, 2014 <https://doi.org/10.1364/OE.22.015982>`_ for detail on the mechanism of RCC. It's a lot slower than AIM, however, it does not
tend to diverge. RCC can only correct x and y coordinates (not 3D).

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

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   Note that the drift is returned first.

   .. code-block:: python

      drift, locs = postprocess.undrift(locs, info, segmentation=1000, display=False)


.. _render-drift-tools:

Inspecting, undoing and reapplying drift
----------------------------------------

The ``Postprocess`` menu (see :doc:`menu-postprocess`) has three more drift
commands:

``Undo drift``
   Undo previous drift correction. Can be pressed again to redo.
``Show drift``
   After drift correction, a drift file is created. If the drift file is
   present, the drift can be displayed with this option.
``Apply drift from an external file``
   Applies drift from a user-specified .txt file. Also supports drag-and-dropping
   the .txt file onto the Render window. Only one file at a time can be processed.

.. note::

   The .txt drift file is automatically saved after each round of drift
   correction. After consecutive rounds, cumulative drift is saved.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   Drift files are read and written with ``io.load_drift`` and
   ``io.save_drift``; applying the negative drift undoes a correction.

   .. code-block:: python

      from picasso import io, postprocess

      drift = io.load_drift("movie_locs_drift.txt")
      locs = postprocess.apply_drift(locs, info, drift=drift)     # apply
      locs = postprocess.apply_drift(locs, info, drift=-drift)    # undo
      fig = postprocess.plot_drift(drift, pixelsize)              # show
