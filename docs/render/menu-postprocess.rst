.. _render-menu-postprocess:

Postprocess Menu
================

.. _render-undrift-by-aim:

Undrift by AIM
--------------

Performs drift correction using the AIM algorithm (Ma, H., et al. Science
Advances. 2024). See :ref:`render-aim`.

.. _render-undrift-from-picked-3d:

Undrift from picked (3D)
------------------------

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

Undo drift (2D)
---------------

Undo previous drift correction (only 2D part). See :ref:`render-drift-tools`.

.. _render-show-drift:

Show drift
----------

Displays the drift after drift correction. See :ref:`render-drift-tools`.

.. _render-apply-drift:

Apply drift from an external file
---------------------------------

Applies drift from a user-specified .txt file. See :ref:`render-drift-tools`.

.. _render-remove-group-info:

Remove group info
-----------------

Removes the group information when loading a dataset that contains group
information. This will, i.e., turn the multicolor representation into a single
color representation.

.. _render-sync-groups:

Sync groups across channels
---------------------------

If more than one channel is present, this function rejects localizations from
across the channels whose *group* field is not present in all channels. This
is useful for removing, for example, clustered localizations after their
cluster centers were filtered with frame analysis.

.. _render-unfold-refold-groups:

Unfold / Refold groups
----------------------

Allows to "unfold" an average to display each structure individually in a
line. Note that the structures need to be grouped and processed with
Picasso: Average beforehand.

.. _render-unfold-groups-square:

Unfold groups (square)
----------------------

Arranges an average in a square so that each structure is displayed
individually. This function does not require Picasso: Average beforehand.
Instead, grouped or picked (circular picks) localizations are accepted.

.. _render-link-localizations:

Link localizations
------------------

Links localizations originating from individual binding events. If the
localizations were already grouped the binding events are never linked across
two input groups.

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

The retained localizations of each binding event are assigned a unique value
in the *group* column. If the localizations were already grouped (e.g., by
picking or clustering), the previous grouping is preserved in the
*group_input* column, and binding events are never linked across two input
groups.

.. _render-align-channels:

Align channels (RCC or from picked)
-----------------------------------

Aligns channels to each other when several datasets are loaded. If picks are
selected, the alignment will be via the center of mass of the picks;
otherwise, an RCC will be used.

.. _render-combine-locs-in-picks:

Combine locs in picks
---------------------

Combines all localizations in each pick to one.

.. _render-apply-expressions:

Apply expressions to localizations
----------------------------------

This tool allows you to apply expressions to localizations, for example:

- ``x +=1`` will shift all localization by one to the right
- ``x +=1; y+=1`` will shift all localization by one to the right and one up.
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

Test clusterer
~~~~~~~~~~~~~~

Checks different clustering parameters on a single pick, see
:ref:`render-test-clusterer`.

Nearest Neighbor Analysis
-------------------------

Calculates distances to the ``k``-th nearest neighbors between two channels,
see :ref:`render-nearest-neighbor-analysis`.

RESI
----

Resolution Enhancement by Sequential Imaging, see :ref:`render-resi`.

Molecular mapping (G5M)
-----------------------

Molecular mapping with G5M, see :ref:`render-g5m`.
