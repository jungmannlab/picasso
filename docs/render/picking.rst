.. _render-picking:

Picking Regions of Interest
===========================

.. _render-pick-shapes:

Pick shapes
-----------

``Picasso: Render`` offers six pick shapes, selected in
``Tools > Tools Settings``. They differ in how they are drawn and in how their
size is set:

.. list-table::
   :widths: 10 30 25 35
   :header-rows: 1

   * - Shape
     - Drawn by
     - Size
     - Suited to
   * - ``Circle``
     - A single left click at its center.
     - ``Diameter``, kept per pick.
     - The default. Compact, roughly round structures. The only shape that ``Pick fiducials`` and ``Subtract pick regions`` work with.
   * - ``Square``
     - A single left click at its center.
     - ``Side length``, kept per pick.
     - The same use as the circle, where a square footprint is preferred.
   * - ``Rectangle``
     - Dragging from one end of its center axis to the other, so it can take any orientation and length. A drag shorter than 5 screen pixels, e.g., a stray click, creates no pick.
     - ``Width`` (across the axis), kept per pick, like the length.
     - Elongated structures - filaments, nanorulers, edges. The only shape for :ref:`Plot pick profile <render-plot-pick-profile>` and the ``x_pick_rot`` / ``y_pick_rot`` columns (see :ref:`Table 1 <files-localization-columns>` of the file formats), which are added when all picks are rectangles.
   * - ``Polygon``
     - One left click per vertex; click the first vertex again to close the outline. A right click removes the last vertex.
     - None; each polygon carries its own extent.
     - Irregular regions that no fixed shape follows, e.g., an outlined cell or organelle.
   * - ``Box``
     - Pressing the left mouse button at one corner, dragging to the opposite one and releasing. A green outline follows the cursor while you drag.
     - None; each box carries its own extent.
     - Quickly grabbing an arbitrary axis-aligned region, without first setting a size.
   * - ``Brush``
     - Painting freehand with the left mouse button held down. Keep painting to extend a region: strokes whose painted areas touch merge into a single pick. They never merge into picks of other shapes.
     - ``Stroke width``, kept per stroke. The setting applies to the next stroke only, so changing it never reshapes what is already painted.
     - Irregular regions that are quicker to sweep over than to outline vertex by vertex.

Removing picks:

- Clicking the right mouse button inside a pick removes that pick, whatever
  its shape. For ``Brush``, this removes the whole painted region under the
  cursor, i.e., all of the strokes merged into it.
- With the ``Polygon`` shape selected, a right click undoes the last step of
  the polygon instead: it removes the last vertex of the last polygon,
  wherever you click, and does nothing if the last pick is not a polygon.
  Select another shape to remove picks by clicking them.
- ``Tools > Clear picks`` (:kbd:`Ctrl+C`) removes all of them.

.. _render-pick-sizes:

Pick sizes and mixed shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every pick keeps the size it was drawn with. The ``Diameter``, ``Side
length`` and ``Width`` in ``Tools > Tools Settings`` set the size of the
*next* picks, like the brush's ``Stroke width``, so picks of different sizes
are drawn by changing the value between picks. To give all picks of a shape
the current value, click ``Apply to all`` next to it.

Picks of different shapes can be mixed as well: changing the shape in
``Tools > Tools Settings`` only changes the shape of the next picks and keeps
the picks drawn so far (only a polygon that is not closed yet is dropped).
With every shape but ``Polygon`` selected, a right click removes the pick
under the cursor, whatever its shape, and the analyses of picked localizations
work on any mix of shapes. Only a few tools need picks of one shape:
``Pick similar`` (see below), ``Subtract pick regions`` (circles) and
``Plot pick profile`` (one rectangle). ``Pick fiducials`` adds circles and
switches the shape to ``Circle``. Saved pick regions keep the shape and size
of each pick (see :ref:`render-save-pick-regions`).

.. _render-picking-steps:

Picking
-------

1. Manual selection. Open ``Picasso: Render`` and load the localization HDF5
   file to be processed.
2. Switch the active tool by selecting ``Tools > Pick``. The mouse cursor will
   now change to a circle. Open ``Tools > Tools Settings`` to change to any of
   the other shapes described above.
3. Set the size of the next picks in the tool settings dialog
   (``Tools > Tools Settings``): ``Diameter`` for circles, ``Side length`` for
   squares, ``Width`` for rectangles or ``Stroke width`` for the brush.
   ``Polygon`` and ``Box`` picks need no size setting. Picks already drawn
   keep their size (see :ref:`render-pick-sizes`).
4. Pick regions of interest by clicking or dragging, as listed in the table
   above. All localizations within the pick will be selected for further
   processing.
5. (Optional, not applicable to ``Polygon`` or ``Brush``) Automated region of
   interest selection. Select ``Tools > Pick similar`` (see
   :ref:`render-pick-similar`) to automatically detect and pick structures
   that have similar numbers of localizations and RMS deviation (RMSD) from
   their center of mass than already-picked structures.

   - The upper and lower thresholds for these similarity measures are the
     respective standard deviations of already-picked regions, scaled by a
     tunable factor. This factor can be adjusted using the field
     ``Tools > Tools Settings > Pick similar ± range``.
   - To display the mean and standard deviation of localization number and
     RMSD for currently picked regions, select ``View > Show info`` and click
     ``Calculate info below``.
   - ``Pick similar`` works with circular, square, rectangular and box picks
     (not with polygon or brush picks, which have no size or canonical form
     to replicate), all of one shape.
   - Circular, square and rectangular picks of different sizes are searched
     for with their median size, which the new picks take; the picks you
     drew keep their sizes.
   - Rectangular picks all take the median length of the already-picked
     regions and are automatically rotated onto the principal axis of the
     localizations they contain, so elongated structures are found at any
     orientation; for them the RMSD along and across that axis are used as
     two separate similarity measures.
   - Box picks likewise all take the median width and height of the
     already-picked regions, while the regions you drew yourself are kept
     exactly as drawn.
   - Without picking any structures first, ``Tools > Pick by localizations
     and RMSD...`` picks circular structures by ranges of the two, chosen
     on a 2D histogram of all candidate picks (see
     :ref:`render-pick-by-n-locs-rmsd`).

6. (Optional) Exporting of pick information. All localizations in picked
   regions can be saved by selecting ``File > Save picked localizations``. The
   resulting HDF5 file will contain a new integer column ``group`` indicating
   to which pick each localization is assigned.
7. (Optional) Statistics about each pick region can be saved by selecting
   ``File > Save pick properties``. The resulting HDF5 file is not a
   localization file. Instead, it holds a dataset called ``groups`` in which
   the rows show statistical values for each pick region. This can also be
   inspected in :doc:`/filter`.
8. (Optional) The picked positions can be saved by
   selecting ``File > Save pick regions``. Such saved pick information can
   also be loaded into ``Picasso: Render`` by selecting
   ``File > Load pick regions`` or by drag-and-dropping it into the Render
   window.

See :doc:`menu-tools` for the other pick utilities.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   Picks are lists of centers in nm (for rectangles, pairs of end
   points), and ``picked_locs`` returns one DataFrame per pick. The pick
   size is the *diameter* of a circular pick (the width for rectangles,
   the side length for squares) in nm, in every function and in the
   pick-region files. ``pick_shape`` and ``pick_size`` take one value for
   all picks or a list with one value per pick, which mixes shapes and
   sizes; ``io.load_picks`` returns them in the same form.

   .. code-block:: python

      import pandas as pd
      from picasso import io, postprocess

      locs, info = io.load_locs("movie_locs.hdf5")

      picks = [(15639.0, 11453.0), (17030.0, 12051.0)]   # centers, nm
      diameter = 100.0                                   # nm
      picked = postprocess.picked_locs(
          locs, info, picks=picks, pick_shape="Circle", pick_size=diameter
      )

      # Pick similar (returns the given picks, followed by the new ones)
      picks = postprocess.pick_similar(
          locs,
          info,
          picks=picks,
          pick_shape="Circle",
          pick_size=diameter,
          std_range=2.0,
      )

      # Save picked localizations (one "group" per pick) and the pick regions
      io.save_locs("movie_locs_picked.hdf5", pd.concat(picked, ignore_index=True), info)
      io.save_picks("movie_locs_picks.yaml", picks, "Circle", size=diameter)

      # Load pick regions saved by Render (positions and diameter in nm)
      picks, shape, diameter = io.load_picks("movie_locs_picks.yaml")

      # Circles of different diameters and a box (no size: None)
      picks = [(15639.0, 11453.0), (17030.0, 12051.0), ((100.0, 100.0), (900.0, 700.0))]
      shapes = ["Circle", "Circle", "Box"]
      sizes = [100.0, 160.0, None]
      picked = postprocess.picked_locs(
          locs, info, picks=picks, pick_shape=shapes, pick_size=sizes
      )
