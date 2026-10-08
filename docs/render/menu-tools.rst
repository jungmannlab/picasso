.. _render-menu-tools:

Tools Menu
==========

The first four entries select the active tool, and ``Tools settings...``
(:kbd:`Ctrl+T`) sets how the tools behave and look; see :doc:`tools`:

- ``Zoom`` (:kbd:`Ctrl+Z`), see :ref:`render-zoom`;
- ``Pick`` (:kbd:`Ctrl+P`), see :ref:`render-pick`;
- ``Measure`` (:kbd:`Ctrl+M`), see :ref:`render-measure`;
- ``Move`` (:kbd:`Ctrl+G`), see :ref:`render-move`;
- ``Tools settings...`` (:kbd:`Ctrl+T`), see :ref:`render-tools-settings`.

The remaining entries work on the picks and the localizations:

.. _render-pick-similar:

Pick similar
------------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+Shift+P`

Automatically identifies picks that are similar to the current picks.
Available for circular, square, rectangular and box picks. For rectangular
picks, the new picks take the median length of the current picks and are
oriented along the localizations they contain. See
:ref:`render-picking-steps` for the similarity measures.

.. _render-remove-locs-in-picks:

Remove localizations in picks
-----------------------------

Remove localizations found in picked region(s) of interest. Can be applied to
separate or all channels simultaneously.

.. _render-move-to-pick:

Move to pick
------------

Changes FoV to display a pick region specified by the user.

.. _render-pick-fiducials:

Pick fiducials
--------------

Automatically picks fiducials. To do so, the whole FOV image is rendered at
one-pixel-blur. Then, such image pixel intensities are histogrammed and the
99th is used as a threshold for selecting image maxima using Localize's
identification.

.. _render-show-trace:

Show trace
----------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+R`

Shows the time trace of the currently selected pick(s).

.. _render-select-picks-trace:

Select picks (trace)
--------------------

Opens a dialog that goes through all picks, displays its trace and asks to
keep or discard it.

.. _render-select-picks-xy:

Select picks (XY scatter)
-------------------------

Opens a dialog that goes through all picks, displays a xy-scatterplot and asks
to keep or discard it.

.. _render-plot-pick-xyz:

Plot pick (XYZ scatter)
-----------------------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+3`

Displays a 3D scatterplot of the localizations of the currently selected
pick(s).

.. _render-select-picks-xyz:

Select picks (XYZ scatter)
--------------------------

Opens a dialog that goes through all picks, displays an xyz-scatterplot and
asks to keep or discard it.

.. _render-select-picks-xyz-4-panels:

Select picks (XYZ scatter, 4 panels)
------------------------------------

Opens a dialog that goes through all picks, displays four panels with an
xyz-scatterplot and a top, bottom and side projection and asks to keep or
discard it.

.. _render-filter-picks-by-count:

Filter picks by count
---------------------

Allows filtering picks by the number of localizations in each pick. When
clicking, a histogram of the number of localizations of all selected picks
will be calculated. A lower and upper boundary can be selected to filter the
picks.

.. _render-clear-picks:

Clear picks
-----------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+C`

Clears all currently selected picks.

.. _render-subtract-pick-regions:

Subtract pick regions
---------------------

Allows loading another pick regions file to subtract from the currently
selected picks. Can be slow for a large number of picks.

.. _render-cluster-in-pick:

Cluster in pick (k-means)
-------------------------

Allows performing k-means clustering in picks. Users can specify the number of
clusters and deselect individual clusters. Picks can be kept or removed. After
looping through all picks an hdf5 file with the cluster information can be
saved.

.. _render-mask-image:

Mask image
----------

Opens a dialog that allows the user to specify a mask for filtering
localizations within and outside it. The user can adjust the histogram bin
size, blur thereof and the threshold applied.

The images can be zoomed in/out (:kbd:`Ctrl`/:kbd:`Cmd` + scrolling) and
panned (dragging with the right mouse button, or with :kbd:`Ctrl`/:kbd:`Cmd` +
the left mouse button). Double clicking resets the zoom.
