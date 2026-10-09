.. _render-picking:

Picking Regions of Interest
===========================

.. _render-pick-shapes:

Pick shapes
-----------

``Picasso: Render`` offers six pick shapes, selected in
``Tools > Tools Settings``. They differ in how they are drawn and in whether
all picks share one size or each pick carries its own extent:

.. list-table::
   :widths: 10 30 25 35
   :header-rows: 1

   * - Shape
     - Drawn by
     - Size
     - Suited to
   * - ``Circle``
     - A single left click at its center.
     - ``Diameter``, shared by all picks.
     - The default. Compact, roughly round structures. The only shape that ``Pick fiducials`` and ``Subtract pick regions`` work with.
   * - ``Square``
     - A single left click at its center.
     - ``Side length``, shared by all picks.
     - The same use as the circle, where a square footprint is preferred.
   * - ``Rectangle``
     - Dragging from one end of its center axis to the other, so it can take any orientation and length. A drag shorter than 5 screen pixels, e.g., a stray click, creates no pick.
     - ``Width`` (across the axis) shared by all picks; the length is per pick.
     - Elongated structures - filaments, nanorulers, edges. The only shape for :ref:`Plot pick profile <render-plot-pick-profile>` and the ``x_pick_rot`` / ``y_pick_rot`` columns (see :ref:`Table 1 <files-localization-columns>` of the file formats).
   * - ``Polygon``
     - One left click per vertex; click the first vertex again to close the outline. A right click removes the last vertex.
     - None; each polygon carries its own extent.
     - Irregular regions that no fixed shape follows, e.g., an outlined cell or organelle.
   * - ``Box``
     - Pressing the left mouse button at one corner, dragging to the opposite one and releasing. A green outline follows the cursor while you drag.
     - None; each box carries its own extent.
     - Quickly grabbing an arbitrary axis-aligned region, without first setting a size.
   * - ``Brush``
     - Painting freehand with the left mouse button held down. Keep painting to extend a region: strokes whose painted areas touch merge into a single pick.
     - ``Stroke width``, kept per stroke. The setting applies to the next stroke only, so changing it never reshapes what is already painted.
     - Irregular regions that are quicker to sweep over than to outline vertex by vertex.

Removing picks:

- Clicking the right mouse button inside a pick removes that pick.
- Two shapes are drawn incrementally and undo their last step instead:
  ``Polygon`` removes the last vertex, and ``Brush`` removes the last stroke,
  wherever you click. Undoing a brush stroke that was joining two painted
  regions leaves them as two separate picks again.
- ``Tools > Clear picks`` (:kbd:`Ctrl+C`) removes all of them.

.. warning::

   The shapes are not interchangeable, so changing the shape while picks exist
   asks for confirmation and then discards them; save them first
   (``File > Save pick regions``) if you want to keep them.

.. _render-picking-steps:

Picking
-------

1. Manual selection. Open ``Picasso: Render`` and load the localization HDF5
   file to be processed.
2. Switch the active tool by selecting ``Tools > Pick``. The mouse cursor will
   now change to a circle. Open ``Tools > Tools Settings`` to change to any of
   the other shapes described above.
3. Set the size of the pick in the tool settings dialog
   (``Tools > Tools Settings``): ``Diameter`` for circles, ``Side length`` for
   squares, ``Width`` for rectangles or ``Stroke width`` for the brush.
   ``Polygon`` and ``Box`` picks need no size setting.
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
     to replicate).
   - Rectangular picks all take the median length of the already-picked
     regions and are automatically rotated onto the principal axis of the
     localizations they contain, so elongated structures are found at any
     orientation; for them the RMSD along and across that axis are used as
     two separate similarity measures.
   - Box picks likewise all take the median width and height of the
     already-picked regions, while the regions you drew yourself are kept
     exactly as drawn.

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
