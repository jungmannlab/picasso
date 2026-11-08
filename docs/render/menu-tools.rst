.. _render-menu-tools:

Tools Menu
==========

The first four entries select the active tool, and ``Tools settings...``
(:kbd:`Ctrl+T`) sets how the tools behave and look; see :doc:`tools`:

- ``Zoom`` (:kbd:`Z`), see :ref:`render-zoom`;
- ``Pick`` (:kbd:`P` or :kbd:`Ctrl+P`), see :ref:`render-pick`;
- ``Measure`` (:kbd:`M` or :kbd:`Ctrl+M`), see :ref:`render-measure`;
- ``Move`` (:kbd:`G` or :kbd:`Ctrl+G`), see :ref:`render-move`;
- ``Tools settings...`` (:kbd:`Ctrl+T`), see :ref:`render-tools-settings`.

The remaining entries work on the picks and the localizations:

.. _render-pick-similar:

Pick similar
------------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+Shift+P`

Automatically identifies picks that are similar to the current picks.
Available for circular, square, rectangular and box picks, all of one shape.
Picks of different sizes are searched for with their median size (diameter,
side length or width), which the new picks take; the picks you drew keep
theirs. For rectangular picks, the new picks take the median length of the
current picks and are oriented along the localizations they contain. New box
picks take the median width and height of the current picks; the boxes you
drew yourself are kept as drawn. See :ref:`render-picking-steps` for the
similarity measures.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``pick_size`` is the diameter for circles, in nm, or one size per pick,
   of which the median is used; ``std_range`` is ``Pick similar ± range``.

   .. code-block:: python

      from picasso import postprocess

      new_picks = postprocess.pick_similar(
          locs,
          info,
          picks=picks,
          pick_shape="Circle",
          pick_size=diameter,
          std_range=2.0,
      )


.. _render-pick-by-n-locs-rmsd:

Pick by localizations and RMSD
------------------------------

Picks similar structures without seed picks: by ranges of the number of
localizations and of the RMSD (the root mean square distance of the
localizations from their center of mass) that a pick contains. Available for
circular picks.

1. Set the pick diameter in ``Tools settings`` (:kbd:`Ctrl+T`) and select
   ``Tools`` > ``Pick by localizations and RMSD...``.
2. The field of view is scanned with candidate picks as in
   :ref:`Pick similar <render-pick-similar>`: each is moved to the center of
   mass of its localizations until it stops moving. The window shows the
   number of localizations and the RMSD of the candidates as a 2D
   histogram, in which structures of one kind form a cluster.

   ``Min. localizations`` leaves out the candidates with fewer
   localizations, by default 1% of the frames. The background forms many
   candidates of a few localizations each, which can outnumber the
   structures. Change it, or the pick diameter, and click
   ``Find candidates`` to scan again.
3. Set the ranges of the structures to pick: draw a rectangle on the
   histogram (:kbd:`Esc` cancels it), or type them in. The number of
   candidates in the ranges is shown next to ``Add picks``.

   ``Find classes`` suggests ranges: it clusters the candidates by their
   number of localizations and RMSD (OPTICS, on both scaled to the same
   spread) and gives each class the ranges of its median ± one standard
   deviation. The ranges of the classes are outlined on the histogram;
   select a class from the list to use its ranges.
4. Click ``Add picks`` to add the candidates in the ranges as circular picks.
   ``File`` > ``Undo`` removes them again.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``pick_similar_candidates`` returns the candidates with their number of
   localizations and RMSD, ``structure_classes`` their classes, and
   ``pick_similar`` takes the ranges instead of seed picks.

   .. code-block:: python

      from picasso import postprocess

      candidates = postprocess.pick_similar_candidates(
          locs, info, pick_size=150, min_n_locs=info[0]["Frames"] // 100
      )
      labels, classes = postprocess.structure_classes(candidates)
      print(classes)

      picks = postprocess.pick_similar(
          locs,
          info,
          pick_size=150,
          n_locs_range=(400, 1000),
          rmsd_range=(35, 50),
      )


.. _render-remove-locs-in-picks:

Remove localizations in picks
-----------------------------

Remove localizations found in picked region(s) of interest. Can be applied to
separate or all channels simultaneously.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   The circle size is the diameter.

   .. code-block:: python

      locs, info = postprocess.remove_locs_in_picks(
          locs, info, picks=picks, pick_shape="Circle", pick_size=diameter
      )


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

1. The whole image is rendered with 100 nm pixels and ``One-Pixel-Blur``.
2. Spots are detected as in Localize's net gradient identification
   (see :ref:`localize-identification`), with a box of about 900 nm and the
   99th percentile of the pixel values as the minimum net gradient.
3. Only spots with more localizations than 80% of the number of frames are
   kept, and each gets a circular pick with a diameter of the box size.

The pick shape in ``Tools > Tools settings`` switches to ``Circle`` with the
box size as the ``Diameter``, so that further picks match the fiducials.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``find_fiducials`` returns the centers and the box size (the pick
   diameter), both in nm. ``undrift_from_fiducials`` picks and undrifts in
   one go.

   .. code-block:: python

      from picasso import imageprocess, postprocess

      picks, box = imageprocess.find_fiducials(locs, info)
      locs, info, drift = postprocess.undrift_from_fiducials(locs, info)


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

- ``X-pos vs frame`` and ``Y-pos vs frame``: the x and y coordinates (nm)
  of each localization;
- ``Localizations``: 1 in frames with a localization, 0 otherwise;
- ``Photons``: the photon count in each frame.

``Export (*.csv)`` in the toolbar saves the trace as ``<name>.trace.csv`` with
three columns: frame, on/off (1 or 0) and photons (as integers).

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      import pandas as pd
      from picasso import lib, postprocess

      picked = postprocess.picked_locs(
          locs, info, picks=picks, pick_shape="Circle", pick_size=diameter
      )
      fig = lib.plot_trace(pd.concat(picked, ignore_index=True), info)


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

Loads another pick regions file and removes the current picks whose centers
lie inside one of its circles. Only for circular picks, both in Render and in
the file; the remaining picks keep their sizes. Can be slow for a large number
of picks.

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
4. Optionally, refine the mask, in this order: ``Fill holes`` fills the areas
   it encloses; ``Min. area (µm²)`` removes the regions smaller than that,
   e.g., specks of background; ``Keep largest regions`` keeps only that number
   of the largest regions (``All`` keeps all), e.g., 1 for the largest cell;
   ``Grow / shrink (nm)`` grows the mask by that distance in all directions,
   or shrinks it for a negative value, e.g., to stay away from the edges of
   the cells (the image border is not an edge). Regions touching at an edge or
   a corner are one region; the panel title shows their number. The settings
   are recorded with the masked localizations.
5. ``Mask`` applies the mask to the localizations and shows those inside it
   (panel *Masked*). Tick ``Mask all channels`` to apply the same mask to every
   loaded channel.
6. ``Save localizations`` saves the localizations inside and outside the mask
   as two files (``_mask_in.hdf5`` and ``_mask_out.hdf5``; with all channels,
   you choose the suffixes). The loaded localizations are not changed.

``Save Mask`` saves the mask as a ``.npy`` array (plus a ``.png`` image) and
``Load Mask`` loads one, e.g., to apply the same mask to another dataset.
``Save Blurred`` saves the blurred image as a ``.png``.

``Add as polygon picks`` adds each region of the mask as a polygon pick, e.g.,
to analyze each cell separately with the spatial statistics (such as
:ref:`render-ripley-h`) or the pick properties, or to save the cells as a
picks file. The outline follows the pixels of the mask, smoothed by up to one
pixel. A region with holes becomes one polygon whose outline runs along a cut
into each hole, around it and back along the cut (a "keyhole" polygon); the
cut adds no area, so the localizations in a hole are outside the pick.
:menuselection:`File --> Undo` removes the added picks.

The panels zoom and pan together: :kbd:`Ctrl`/:kbd:`Cmd` + scrolling zooms,
dragging with the right mouse button (or :kbd:`Ctrl`/:kbd:`Cmd` + the left
button) pans, and a double click resets the zoom.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``mask_image`` takes a method name or a threshold between 0 and 1
   (``Custom``). The mask spans the whole field of view. ``mask_locs``
   returns the metadata of both parts, with the area inside or outside the
   mask under ``Area (um^2)`` (used, e.g., by SPINNA) and ``mask_info``.
   To mask other localizations, e.g., another channel, with the same mask,
   pass the field of view of the first as ``field_of_view``.
   ``label_components``, ``keep_components``, ``fill_holes``, ``erode`` and
   ``dilate`` refine a mask; lengths are in the unit of ``pixel_size``, e.g.,
   nm, and areas in its square. ``mask_to_polygons`` outlines each region as
   a polygon pick (``tolerance`` in nm, 0 for the exact outline of the
   pixels, which holds the same localizations as ``mask_locs``).

   .. code-block:: python

      import numpy as np
      from picasso import io, lib, masking

      locs, info = io.load_locs("movie_locs.hdf5")
      image = masking.generate_image(locs, info, disp_px_size=300, blur=500)   # nm
      mask, threshold = masking.mask_image(image, "otsu")   # or 0.5, "li", "local_mean", ...
      mask = masking.fill_holes(mask)
      mask = masking.keep_components(mask, n_largest=1)       # the largest cell
      mask = masking.erode(mask, 500.0, pixel_size=300.0)    # nm
      locs_in, info_in, locs_out, info_out = masking.mask_locs(
          locs, info, mask=mask, mask_info={"Blur (nm)": 500}
      )

      io.save_locs("movie_locs_mask_in.hdf5", locs_in, info_in)
      io.save_locs("movie_locs_mask_out.hdf5", locs_out, info_out)
      np.save("movie_locs_mask.npy", mask)

      # each region (cell) as a polygon pick
      picks = masking.mask_to_polygons(mask, field_of_view=lib.field_of_view(info))
      io.save_picks("movie_locs_mask_picks.yaml", picks, "Polygon")
