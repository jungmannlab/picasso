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
Available for circular, square, rectangular and box picks. New circular and
square picks take the current size. For rectangular picks, the new picks take
the median length of the current picks and are oriented along the
localizations they contain. New box picks take the median width and height of
the current picks; the boxes you drew yourself are kept as drawn. See
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

Automatically picks fiducial markers, e.g., for
:ref:`marker-based drift correction <render-marker-drift>`. Fiducials are
bright spots that are visible in nearly every frame, so they appear as the
densest spots of the image. Requires no existing picks.

1. The whole image is rendered with one pixel per camera pixel and
   ``One-Pixel-Blur``.
2. Spots are detected as in Localize's net gradient identification
   (see :ref:`localize-identification`), with a box of about 900 nm and the
   99th percentile of the pixel values as the minimum net gradient.
3. Only spots with more localizations than 80% of the number of frames are
   kept, and each gets a circular pick with a diameter of the box size.

.. _render-plot-pick-profile:

Plot pick profile
-----------------

Plots the distribution of the localizations along a single rectangular pick, as a histogram of their
positions along the pick's axis in nm. Requires exactly one rectangular pick
(see :ref:`render-pick-shapes`). With several channels, they can be plotted
together, each in its own color.

- ``Bin width`` in the toolbar sets the histogram bin width.
- ``Export (*.csv)`` saves the positions (nm), one column per channel.

.. _render-show-trace:

Show trace
----------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+R`

Plots the localizations of the picks against time, e.g., to check binding
kinetics or to tell repeated binding from a single sticking event. With
several picks, their localizations are combined into one trace. The window
has four panels, all over the frames of the movie:

- ``X-pos vs frame`` and ``Y-pos vs frame``: the x and y coordinates (camera
  pixels) of each localization;
- ``Localizations``: 1 in frames with a localization, 0 otherwise;
- ``Photons``: the photon count in each frame.

``Export (*.csv)`` in the toolbar saves the trace as ``<name>.trace.csv`` with
three columns: frame, on/off (1 or 0) and photons (as integers).

.. _render-select-picks-trace:

Select picks (trace)
--------------------

Goes through the picks one by one, shows the trace of each (the same panels
as in :ref:`render-show-trace`) and asks whether to keep it: ``Accept`` keeps
the pick, ``Reject`` removes it, ``Back`` returns to the previous pick and
``Cancel`` stops. The dialog shows the progress, the number of kept and
removed picks and the time per pick.

.. _render-select-picks-xy:

Select picks (XY scatter)
-------------------------

Opens a dialog that goes through all picks, displays a xy-scatterplot and asks
to keep or discard it.

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

Splits the localizations into those inside and outside a mask, e.g., to keep
the localizations in a cell and remove the background. The mask is computed
from the density of the localizations of the selected channel:

1. The localizations are histogrammed with ``Display pixel size (nm)``
   (300 nm by default) over the whole image and normalized to a maximum of 1
   (panel *Histogrammed localizations*).
2. The histogram is blurred with a Gaussian of :math:`\sigma` =
   ``Blur (nm)`` (500 nm by default) and normalized again (panel *Blur*).
   ``Show histogram`` plots the pixel values of the blurred image, which helps
   to choose a threshold.
3. Pixels above the threshold form the mask (panel *Mask*). ``Custom`` uses the
   ``Threshold`` typed in (0 to 1, 0.5 by default). The other methods find it
   automatically with the scikit-image functions of the same name: global
   thresholds (``Isodata``, ``Li``, ``Mean``, ``Minimum``, ``Otsu``,
   ``Triangle``, ``Yen``) or local ones that vary across the image
   (``Local Gaussian``, ``Local mean``, ``Local median``).
4. ``Mask`` applies the mask to the localizations and shows those inside it
   (panel *Masked*). Tick ``Mask all channels`` to apply the same mask to every
   loaded channel.
5. ``Save localizations`` saves the localizations inside and outside the mask
   as two files (``_mask_in.hdf5`` and ``_mask_out.hdf5``; with all channels,
   you choose the suffixes). The loaded localizations are not changed.

``Save Mask`` saves the mask as a ``.npy`` array (plus a ``.png`` image) and
``Load Mask`` loads one, e.g., to apply the same mask to another dataset.
``Save Blurred`` saves the blurred image as a ``.png``.

The panels zoom and pan together: :kbd:`Ctrl`/:kbd:`Cmd` + scrolling zooms,
dragging with the right mouse button (or :kbd:`Ctrl`/:kbd:`Cmd` + the left
button) pans, and a double click resets the zoom.
