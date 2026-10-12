.. _render-menu-postprocess:

Postprocess Menu
================

.. _render-undrift-by-aim:

Undrift by AIM
--------------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+U`

Performs drift correction using the AIM algorithm (Ma, H., et al. Science
Advances. 2024). See :ref:`render-aim`.

.. _render-undrift-from-picked-3d:

Undrift from picked
-------------------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+Shift+U`

Performs drift correction using the picked localizations as fiducials, also in
z if the dataset has 3D information. See :ref:`render-marker-drift`.

.. _render-undrift-from-picked-2d:

Undrift from picked (2D)
------------------------

Performs drift correction using the picked localizations as fiducials, without
z even if the dataset has 3D information. See :ref:`render-marker-drift`.

.. _render-undrift-by-rcc:

Undrift by RCC
--------------

Performs drift correction by redundant cross-correlation. See
:ref:`render-rcc`.

.. _render-undo-drift:

Undo drift
----------

Undo previous drift correction. See :ref:`render-drift-tools`.

.. _render-show-drift:

Show drift
----------

Displays the drift after drift correction. See :ref:`render-drift-tools`.

.. _render-apply-drift:

Apply drift from an external file
---------------------------------

Applies drift from a user-specified .txt file. See :ref:`render-drift-tools`.

.. _render-remove-columns:

Remove columns
--------------

Removes the selected columns from the localizations of a channel, e.g., the
``group`` column, so that a single file is drawn with the colormap again
instead of one color per group (see :ref:`render-coloring`). The required
columns (``frame``, ``x``, ``y``, ``lpx``, ``lpy`` and, for 3D data, ``z`` and
``lpz``) cannot be removed. The removed columns are recorded in the metadata;
save the localizations to keep the change.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      locs = locs.drop(columns=["group"])


.. _render-sync-groups:

Synchronize groups across channels
----------------------------------

If more than one channel is present, this function rejects localizations from
across the channels whose *group* field is not present in all channels. This
is useful for removing, for example, clustered localizations after their
cluster centers were filtered with frame analysis.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      from picasso import lib

      locs_ch1, locs_ch2 = lib.sync_groups([locs_ch1, locs_ch2])


.. _render-unfold-groups-square:

Unfold groups/picks (square grid)
---------------------------------

Arranges the groups (or picks) side by side on a square grid, so that many
structures can be viewed and compared at once. The user selects the number of elements per column and the
distance between them (250 nm by default).

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``spacing`` is in camera pixels.

   .. code-block:: python

      locs, info = lib.unfold_localizations_square(
          locs, info, n_square=10, spacing=250 / pixelsize
      )


.. _render-link-localizations:

Link localizations (binding events)
-----------------------------------

Merges the localizations of one binding event (one molecule seen in
consecutive frames) into a single localization. Localizations are linked when
they are closer than ``Max. distance (nm)`` (10 nm by default) and separated by
at most ``Max. transient dark frames`` (3 by default) frames without a
localization, which bridges short blinks.

Each linked localization has the precision-weighted mean position, the summed
photons and the combined (smaller) localization precision of its event, plus
the columns ``len`` (duration of the event in frames, gaps included) and ``n``
(number of localizations merged); see :ref:`Table 1 <files-localization-columns>`.

If the localizations have a ``group`` column (e.g., after picking or
clustering), only localizations of the same group are linked: two nearby
localizations from different groups always stay separate events.

With several channels loaded, choose *Apply to all sequentially* to link every
channel with the same parameters. Channels that are already linked are skipped.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``r_max`` is in camera pixels. ``compute_dark_times`` adds the ``dark``
   column to linked localizations.

   .. code-block:: python

      from picasso import io, lib, postprocess

      locs, info = io.load_locs("movie_locs.hdf5")
      pixelsize = lib.get_from_metadata(info, "Pixelsize")

      linked = postprocess.link(locs, info, r_max=10 / pixelsize, max_dark_time=3)
      linked = postprocess.compute_dark_times(linked)
      print(linked["len"].mean(), linked["dark"].mean())   # frames


.. _render-select-central-frames:

Select central frames localizations
-----------------------------------

Groups localizations into binding events, exactly like *Link localizations*
(same dialog: maximum distance and maximum number of transient dark frames).
Instead of merging each binding event into a single localization, it keeps the
localizations that are not at the borders of the event, i.e., those in the
first and the last frame of each event are discarded, see
`Steen et al., Nature Methods 21, 1755-1762 (2024)
<https://doi.org/10.1038/s41592-024-02374-8>`_, Extended Data Fig. 1f.

The retained localizations of each binding event get the same value in the
``group`` column, one value per event. If the localizations already had a
``group`` column (e.g., after picking or clustering), it is moved to
``group_input``, and only localizations of the same input group are joined
into an event: two nearby localizations from different picks or clusters
always end up in separate events.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      cores = postprocess.select_binding_event_cores(
          locs, r_max=10 / pixelsize, max_dark_time=3, min_n_locs=3
      )


.. _render-align-channels:

Align channels (RCC or from picked)
-----------------------------------

Aligns channels to each other when several datasets are loaded. If picks are
selected, the alignment will be via the center of mass of the picks;
otherwise, redundant cross-correlation (RCC, see :ref:`render-rcc`) will be
used (always 2D only).

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   The circle size of ``align_from_picked`` is the diameter.

   .. code-block:: python

      aligned = postprocess.align([locs_ch1, locs_ch2], [info_ch1, info_ch2])   # RCC
      aligned = postprocess.align_from_picked(
          [locs_ch1, locs_ch2], [info_ch1, info_ch2],
          picks=picks, pick_shape="Circle", pick_size=2 * radius,
      )


.. _render-stitch-tiles:

Stitch tiles
------------

Stitches tiles recorded at different stage positions into one dataset. A
large field of view is often imaged as a grid of tiles, all with the same
camera, so the localizations of every tile are in the same camera frame: loaded
together, the tiles lie on top of each other, although neighboring tiles only
share a few microns. Stitching translates each tile to its place and merges
the tiles.

#. Load the tiles (one file per tile and target) and run ``Postprocess`` >
   ``Stitch tiles...``.
#. The table lists the loaded channels with their stage positions, read from
   the Micro-Manager metadata (``XPositionUm`` and ``YPositionUm``). Channels
   recorded at the same stage position, e.g., the targets of an
   Exchange-PAINT experiment, get the same **Tile** number and are aligned
   together. The **Target** number says which channels of different tiles
   are merged into one channel (by default, the n-th channel of each tile).
   Untick **Align** to leave a channel out of the alignment, e.g., a sparse
   target.
#. Choose the **Initial positions**:

   - *Stage positions*: from the metadata. The stage axes may be flipped or
     swapped relative to the camera axes; *Automatic* **Stage orientation**
     tries all eight possibilities and takes the one whose overlaps match
     best.
   - *Current positions*: where the channels are now, e.g., after placing
     them roughly with the :ref:`Move tool <render-move>`. **Use this if the
     files contain no stage positions.**

#. Click ``OK``. Each pair of overlapping tiles is matched by
   cross-correlating the overlap within the **Search radius** (the stage
   positions are typically off by a few microns) and refined to sub-pixel
   precision by pairing nearest neighbors. The offsets of all pairs are
   combined, and pairs that disagree with the others are dropped.

The tiles are replaced by one stitched channel per target, named
``<first tile>_stitched.hdf5``; save them with ``File`` > ``Save
localizations``. A summary lists the stage orientation found and, for each
pair of overlapping tiles, whether it was used and how well it agrees with
the final positions (residual, camera pixels).

Options:

- **Correct z offsets** (3D data): each tile is also shifted in z to match
  its neighbors in the overlaps.
- **Cut overlaps at the seams**: each overlap keeps the localizations of only
  the tile whose center is closest, so no molecule is counted twice.
  Otherwise, overlaps contain the localizations of all tiles at double
  density.
- **Offset frames per tile**: the frames of each tile are offset by the
  frames of the tiles acquired before it (in the order of the Micro-Manager
  acquisition times), so frames stay unique.

.. note::

   Stitching assumes that the tiles are only translated relative to each
   other, which holds for one camera at different stage positions. Undrift
   the tiles before stitching.

Also available as the command :ref:`picasso stitch <cmd-stitch>`.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``stitch.stitch`` returns one localization table and metadata per target,
   and a report with the tile positions and the pairwise offsets.

   .. code-block:: python

      from picasso import io, stitch

      paths = ["tile1.hdf5", "tile2.hdf5", "tile3.hdf5", "tile4.hdf5"]
      locs, infos = zip(*[io.load_locs(path) for path in paths])
      stitched, stitched_infos, report = stitch.stitch(
          list(locs), list(infos), search_radius_um=10, paths=paths
      )
      io.save_locs("stitched.hdf5", stitched[0], stitched_infos[0])
      print(report["orientation"], report["positions"])


.. _render-combine-locs-in-picks:

Combine localizations in picks
------------------------------

Replaces the localizations of each pick with a single localization. Internally, all localizations of a pick are linked into one event, as in :ref:`render-link-localizations` but without any
distance or dark-time limit.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   The circle size is the diameter.

   .. code-block:: python

      combined = postprocess.combine_locs_in_picks(
          locs, info, picks=picks, pick_shape="Circle", pick_size=2 * radius
      )


.. _render-apply-expressions:

Apply expression to localizations
---------------------------------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+A`

This tool allows you to apply expressions to localizations, for example:

- ``x +=1`` will shift all localization by one camera pixel to the right
- ``x +=1; y+=1`` will shift all localization by one camera pixel to the right and one up.
- ``flip x z`` will exchange the x-axis with y-axis if z localizations are
  present (side projection), similar for ``flip y z``.
- ``spiral r n`` will plot each localization over the time of the movie in a
  spiral with radius r and n number of turns (e.g., to detect repetitive
  binding), ``uspiral`` to reverse.

.. note::

   Using two variables in one statement is not supported (e.g. ``x = y``). To
   filter localizations use Picasso: Filter.

Localizations moved outside the image by an expression are kept: the canvas is
fitted to them as described for the Move tool (see :ref:`render-move`).
Invalid localizations (e.g., NaN or negative localization precision) are
removed.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   Expressions are plain column arithmetic on the DataFrame. ``z`` is in nm
   while ``x`` and ``y`` are in camera pixels, so a flip converts.

   .. code-block:: python

      locs["x"] += 1
      locs["y"] += 1
      locs["x"], locs["z"] = locs["z"] / pixelsize, locs["x"] * pixelsize   # flip x z


.. _render-menu-clustering:

Clustering
----------

The ``Clustering`` submenu holds the clustering algorithms, described in
:ref:`render-clustering`.

DBSCAN
~~~~~~

Cluster localizations with the dbscan clustering algorithm, see
:ref:`render-dbscan`.

HDBSCAN
~~~~~~~

Cluster localizations with the hdbscan clustering algorithm, see
:ref:`render-hdbscan`.

SMLM clusterer
~~~~~~~~~~~~~~

Cluster localizations with the custom algorithm designed for SMLM, see
:ref:`render-smlm-clusterer`.

Test clustering
~~~~~~~~~~~~~~~

Checks different clustering parameters on a single pick, see
:ref:`render-test-clusterer`.

Calculate nearest neighbor distances
------------------------------------

Calculates distances to the ``k``-th nearest neighbors between two channels,
see :ref:`render-nearest-neighbor-analysis`.

Coordinate-based colocalization (CBC)
-------------------------------------

Scores the colocalization of each localization of one channel with another
channel, for one pair or all pairs of channels, see :ref:`render-cbc`.

RESI
----

Resolution Enhancement by Sequential Imaging, see :ref:`render-resi`.

Molecular mapping (G5M)
-----------------------

Molecular mapping with G5M, see :ref:`render-g5m`.
