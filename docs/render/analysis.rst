.. _render-analysis:

Clustering and Molecular Mapping
================================

This page describes RESI and G5M for resolving and mapping individual molecules, several clustering algorithms, a nearest neighbor analysis, coordinate-based colocalization and spatial statistics functions. All of them are found in the ``Postprocess`` menu (see :doc:`menu-postprocess`).

.. _render-resi:

RESI
----

.. figure:: /images/render_resi.png
   :width: 374px
   :alt: Picasso Render RESI dialog with per-channel clustering parameters

   The RESI dialog.

RESI (Resolution Enhancement by Sequential Imaging) is a derivative technique of DNA-PAINT where the same target species is stochastically labeled with different DNA strands, allowing for a substantial resolution boost (`Reinhardt, Masullo, Baudrexel, Steen, et al., Nature, 2023 <https://doi.org/10.1038/s41586-023-05925-9>`__).

To use RESI:

1. Prepare your individual RESI channels (localization, drift correction, filtering
   and **alignment**).
2. Load all such localization lists into Picasso Render and open
   ``Postprocess > RESI``. The dialog shown above will appear.
3. Each channel will be clustered using the SMLM clusterer (molecular mapping methods could be used potentially as well). Clustering parameters can be defined for
   each RESI channel individually, although it is possible to apply the same
   parameters to all channels by clicking
   ``Apply the same clustering parameters to all channels``, which will copy
   the clustering parameters from the first row and paste it to all other
   channels.
4. Next, the user needs to specify whether or not to save clustered
   localizations or cluster centers from each of the RESI channels
   individually, and whether to apply basic frame analysis (to minimize the
   effect of sticking events). For the explanation of the parameters, see
   :ref:`render-smlm-clusterer`.
5. Upon clicking ``Perform RESI analysis``, each of the loaded channels is
   clustered, cluster centers are extracted and combined from all RESI
   channels to create the final RESI file under the name specified by the user.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   The channels must be aligned already. Radii are in nm and may be given
   per channel as lists.

   .. code-block:: python

      from picasso import io, postprocess

      paths = ["round1_locs.hdf5", "round2_locs.hdf5"]
      locs, infos = zip(*[io.load_locs(p) for p in paths])

      resi_locs, resi_info = postprocess.resi(
          list(locs), list(infos),
          radius_xy=10.0,                 # nm, e.g., 2 x NeNA
          min_locs=10,
          apply_fa=True,                  # basic frame analysis
          progress_callback="console",
      )
      io.save_locs("resi.hdf5", resi_locs, resi_info)


.. _render-g5m:

G5M
---

G5M (Gaussian Mixture Modeling with Modifications for Molecular Mapping) is a method for finding molecules' positions from DNA-PAINT localizations (`Kowalewski, Reinhardt, et al., Nature Communications, 2026 <https://doi.org/10.1038/s41467-026-70198-5>`__). It is opened via ``Postprocess > Molecular mapping (G5M)``.

G5M is based on Gaussian Mixture Modeling (GMM) but includes several
modifications to make it suitable for molecular mapping in DNA-PAINT. All the
technicalities as well as the user guide of the method are explained in the
publication mentioned and its Supplementary Information. Please refer to
:mod:`picasso.g5m` for the details of the implementation. Below is a brief
summary of the user guide.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   G5M needs clustered localizations (a ``group`` column), e.g., from the
   DBSCAN. For 3D astigmatism data pass
   ``calibration=io.load_calibration(...)``; the other keyword arguments
   are the dialog's settings.

   .. code-block:: python

      from picasso import clusterer, g5m, io

      locs, info = io.load_locs("movie_locs.hdf5")

      clustered, _ = clusterer.dbscan(
          locs, radius=20.0, min_samples=4,   # radius in nm
      )
      molecules, clustered, info = g5m.g5m(
          clustered, info, min_locs=10, callback_parent="console"
      )
      io.save_locs("movie_molmap.hdf5", molecules, info)


.. _render-g5m-preprocessing:

Preprocessing
~~~~~~~~~~~~~

G5M requires some preprocessing of localizations to filter out the badly
fitted ones, especially the ones arising from crosstalk (overlapping
blinking). The most robust approach is to inspect the localizations for ``log_likelihood`` or ``chi_square`` using :ref:`rendering by property <render-properties>` to then filter out the low-quality fits. Additionally one can:

- These can be excluded from 2D data where the ellipticity and size of the
  image of an emitter in x and y can be filtered (in Picasso these are found
  under names "ellipticity", "sx" and "sy", respectively).
- Moreover, the photon count can be cut-off as crosstalk is likely to result
  in a higher-intensity signal.
- In 3D data these 2D filters can be less reliable (PSF engineering).

We strongly encourage avoiding dense blinking, where emission signals from
neighboring molecules overlap, especially during 3D image acquisition.

Note that G5M assumes that the localization precision values (``lpx``,
``lpy`` and ``lpz`` columns) correspond to the real spread of the localization
clouds. For example, drift correction needs to be done precisely. In astigmatic
imaging, great care needs to be taken if fiducials are used for drift
correction, especially if they lie at a different plane from the target
localizations. In fact, we recommend using the new fiducial-free algorithms
such as `COMET <https://www.biorxiv.org/content/10.64898/2026.03.27.714864v1>`_
and `AIM <https://www.science.org/doi/10.1126/sciadv.adm7765>`_ (see
:ref:`render-aim`). This prevents overfitting of too many molecules, i.e., falsely detecting molecules that do not exist.

.. _render-g5m-clustering:

Clustering before G5M
~~~~~~~~~~~~~~~~~~~~~

Prior to molecular mapping, clustering of localizations is required to split
the data into smaller chunks. For many datasets, :ref:`DBSCAN <render-dbscan>`
works well. While in some cases some adjustments may be needed, we recommend
the following DBSCAN parameters:

- Radius (epsilon): :math:`2 \times \mathrm{LP}` in 2D and
  :math:`3 \times \mathrm{LP}` in 3D, where LP is the average localization
  precision of the dataset (e.g., :ref:`NeNA <render-nena>` or the median
  localization precision).
- Min. samples: 4 (the default).

Clustering in Picasso adds the ``group`` column to the localization file,
which is required for G5M.

.. important::

   G5M uses the information from the ``group`` column to split localizations into manageable chunks. ``group_input`` can also be used, so if localizations are picked after DBSCAN, G5M can still process it. Both columns are described in :ref:`Table 1 <files-localization-columns>` of the file formats.

Localizations obtained with the rotated elliptical Gaussian model will
automatically take the mean value of the xy-plane rotation and apply it to the
fitted molecule.

.. _render-g5m-postprocessing:

Postprocessing
~~~~~~~~~~~~~~

To account for fluorophore non-specific sticking, frame analysis is normally
recommended (especially the filtering of st. dev. of frame per molecule).
However, if localizations from neighboring localization clouds overlap, this
is not sufficient due to ambiguous assignment of localizations to molecules.
Therefore, we recommend filtering of molecules that express too few binding
events (saved in the column ``n_events``) on top of the ``std_frame`` filtering. In the publication, we recommend a threshold of at least 3 binding events per molecule.

The final postprocessing step is log-likelihood filtering (using the column
``p_val``). The threshold used in the publication was ``> 0.0015``, however, it might need
to be adjusted for your data, especially in 3D this can be too conservative. Higher thresholds might be used for more aggresive filtering if overfitting is too strong, however, real molecules may be rejected.

.. _render-g5m-overfitting:

Checking for overfitting
~~~~~~~~~~~~~~~~~~~~~~~~

As a final check for overfitting (i.e., too many assigned molecules), G5M
saves the following automatically:

- **Binding events per molecule.** A bar plot of the number of binding events
  per molecule (``n_events`` column) for clustered (with neighbors within 25
  nm) and sparse (without neighbors within 80 nm) molecules. If the clustered
  molecules show fewer binding events than the sparse molecules, overfitting
  likely occurred. See Fig. S15 of the publication for an example of
  well-behaved data. The same test can be rerun with other distances in
  Picasso Filter, see :ref:`filter-test-subclustering`.
- **Relative σ values**, i.e., the fitted Gaussian σ divided by
  the average loc. precision around the molecule. This can be used to estimate
  if the loc. precision values are accurate (if not, many molecules will have
  relative σ values close to the user-selected min./max. σ).
- **KS 2 sample test**, comparing the two
  distributions. The output test statistic and theoretical p value correspond
  to the KS test, while the permutation p value is calculated by randomly
  permuting the labels of clustered and sparse molecules 1,000 times; see
  :ref:`filter-test-subclustering` for the details.

.. _render-g5m-troubleshooting:

Troubleshooting
~~~~~~~~~~~~~~~

If the outcome of G5M seems unsatisfactory, please check the following:

- Make sure that ``group`` column is present in the localization file and
  contains the correct information (i.e., from DBSCAN clustering, not from
  picking localizations). ``group_input`` can also be used.
- (3D) Consider using a more accurate fitting model, such as spline fitting; we
  found that combining it with an optional increase in max. sigma (for example 3) can remove the overfitting problem, especially when combined with filtering afterwards (see :ref:`render-g5m-postprocessing`).
- Make sure that the loc. precision values (columns ``lpx``, ``lpy``,
  ``lpz``) are correct, comparing NeNA and median loc. precision is a
  reasonable proxy (without fiducial markers); problems may arise due to a
  miscalibrated camera, leading to incorrect photon counts and thus incorrect
  loc. precisions or using out-of-focus gold fiducials for drift correction. Misaligned microscope setup can introduce aberrations not modeled during localization.
- Another reason why the loc. precision values can be off is due to the small
  box size in the localization step; for example in 3D astigmatic imaging,
  single-emitter images can be quite large, potentially exceeding the
  user-defined box size; in such cases, we recommend increasing the box size
  in the localization step and rerunning the analysis, see :ref:`Box side length <localize-box-size>`.
- Inspect if the localizations were preprocessed as described in
  :ref:`render-g5m-preprocessing`.
- Rerun G5M without postprocessing (filtering) and redo it manually,
  since some steps may be too stringent, such as ``p_val`` or ``n_events``
  (latter especially for short acquisition times).
- Adjust min./max. σ, especially too low max. σ may lead to high false
  positive error rates (i.e., overfitting).

  - We suggest inspecting ``rel_sigma`` values of the assigned molecules,
    which are calculated as the fitted σ divided by the mean localization
    precision of the surrounding localizations.
  - If the values are close to the user-selected min./max. σ, min./max. σ
    might need to be adjusted. Alternatively, this might be a sign of
    inaccurate/imprecise loc. precision values, see above.

- Adjust min. locs.
- Adjust DBSCAN (or other clustering algorithm) parameters. For example, if
  G5M takes too long to run, at least one DBSCAN cluster contains too many
  molecules. In such a case, we recommend splitting such clusters further.

.. _render-clustering:

Clustering
----------

The clustering algorithms are found in the ``Postprocess > Clustering``
submenu.

.. tip::

   It is recommended to remove any fiducial markers before clustering
   (with any of the algorithms), to lower clustering time, given they are of
   no interest to the user. To do that, the markers can be picked and removed
   using ``Tools > Remove localizations in picks``.

.. _render-dbscan:

DBSCAN
~~~~~~

Clusters localizations with scikit-learn's `DBSCAN
<https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_
(Ester et al., KDD, 1996). A localization with at least ``Min. samples``
localizations (itself included) within ``Radius`` is a core point. Core points
within ``Radius`` of each other form one cluster, together with the
localizations within ``Radius`` of them. All other localizations are noise and
are removed.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Picasso
     - scikit-learn
     - Meaning
   * - ``Radius (nm)``
     - ``eps``
     - Neighborhood radius.
   * - ``Radius z (nm)`` (3D only)
     - none
     - Neighborhood radius in z. The z coordinates are scaled so that the
       neighborhood is an ellipsoid with semi-axes (radius, radius, radius z).
   * - ``Min. samples``
     - ``min_samples``
     - Localizations within the radius for a core point.
   * - ``Min. no. of locs``
     - none
     - Clusters with fewer localizations are removed afterwards.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   Radii are in nm (``radius_z`` for 3D data).

   .. code-block:: python

      from picasso import clusterer, io

      locs, info = io.load_locs("movie_locs.hdf5")

      clustered, cluster_info = clusterer.dbscan(
          locs, radius=20.0, min_samples=10, min_locs=10
      )
      io.save_locs("movie_locs_dbscan.hdf5", clustered, info + [cluster_info])


.. _render-hdbscan:

HDBSCAN
~~~~~~~

Clusters localizations with scikit-learn's `HDBSCAN
<https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html>`_
(`Campello et al., PAKDD, 2013
<https://doi.org/10.1007/978-3-642-37456-2_14>`__). HDBSCAN runs DBSCAN over
all radii at once and keeps the clusters that persist over the widest range
of them. It needs no radius and finds clusters of different densities.
Localizations outside the clusters are noise and are removed. In 3D, x, y and
z are treated alike (no separate z radius).

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Picasso
     - scikit-learn
     - Meaning
   * - ``Min. cluster size``
     - ``min_cluster_size``
     - Smallest group of localizations kept as a cluster.
   * - ``Min. samples``
     - ``min_samples``
     - Localizations in a neighborhood for a core point. Larger values make
       the clustering more conservative: more localizations end up as noise.
   * - ``Intercluster max. distance (nm)``
     - ``cluster_selection_epsilon``
     - Clusters closer than this distance are merged; 0 (the default) turns
       merging off.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      clustered, cluster_info = clusterer.hdbscan(
          locs, min_cluster_size=10, min_samples=10
      )
      io.save_locs("movie_locs_hdbscan.hdf5", clustered, info + [cluster_info])


.. _render-smlm-clusterer:

SMLM clusterer
~~~~~~~~~~~~~~

This algorithm is for finding localizations corresponding to individual molecules in DNA-PAINT since it assumes they cluster as spherical Gaussian. The name "SMLM clusterer" is not very accurate and may be changed in one of future releases.

In short, localizations with the maximum number of neighboring localizations within a
user-defined radius are chosen as cluster centers, around which all
localizations within the given radius belong to one cluster. If two or more
local maxima are within the radius, the clusters are merged. The algorithm
was used in `Schlichthaerle, Lindner and Jungmann, Nature Communications, 2021
<https://doi.org/10.1038/s41467-021-22606-1>`__ and `Reinhardt, Masullo,
Baudrexel, Steen, et al., Nature, 2023
<https://doi.org/10.1038/s41586-023-05925-9>`__.

SMLM clusterer requires three (or four if 3D data is processed) arguments:

Radius
   Final size of the clusters.
Radius z (3D only)
   Final size of the clusters in the z axis. If the value is different from
   radius in xy plane, clusters have ellipsoidal shape. Radius z can have a
   different value to account for a difference in localization precision in
   lateral and axial directions.
Min. locs
   Minimum number of localizations in a cluster.
Basic frame analysis
   If True, each cluster is checked for its value of mean frame (if it is
   within the first or the last 20% of the total acquisition time, it is
   discarded). Moreover, localizations inside each cluster are split into 20
   time bins (across the whole acquisition time). If a single time bin
   contains more than 80% of localizations per cluster, the cluster is
   discarded.

.. note::

   Alternatively, filter out the cluster centers with low ``std_frame``, the
   standard deviation of the frames of the localizations in each cluster
   (molecule). Sticking events are short, so their localizations span only a
   few frames, while repetitive binding to a molecule is spread over the whole
   acquisition. G5M, for example, removes molecules with ``std_frame`` below
   10% of the number of frames.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``radius_xy`` and ``radius_z`` are in nm; ``frame_analysis`` is the
   basic frame analysis. ``find_cluster_centers`` gives one localization per
   cluster.

   .. code-block:: python

      clustered, cluster_info = clusterer.cluster(
          locs, radius_xy=20.0, min_locs=10, frame_analysis=True,
          radius_z=None, progress="console",
      )
      centers = clusterer.find_cluster_centers(clustered, progress="console")
      io.save_locs("movie_locs_clusters.hdf5", clustered, info + [cluster_info])
      io.save_locs("movie_locs_cluster_centers.hdf5", centers, info + [cluster_info])


.. _render-test-clusterer:

Test clustering
~~~~~~~~~~~~~~~

``Postprocess > Clustering > Test clustering...`` tries clustering
parameters on a small region before the whole dataset is clustered:

1. Pick exactly one region of interest in the main window (see
   :ref:`render-picking`).
2. In the dialog, select the channel and the algorithm: ``DBSCAN``,
   ``HDBSCAN``, ``SMLM`` (the :ref:`SMLM clusterer <render-smlm-clusterer>`)
   or ``G5M``. The parameters are those of the respective dialogs, see above
   and :ref:`render-g5m`. ``G5M`` first runs DBSCAN (``DBSCAN radius``,
   ``DBSCAN min. samples``) and then G5M on the DBSCAN clusters. For 3D data,
   the z parameters are shown, and for G5M the 3D fit mode is detected from
   the metadata; astigmatic data need ``Load 3D calibration``.
3. Click ``Test``. The localizations in the pick are clustered and rendered in
   the ``View`` box. Change the parameters and click ``Test`` again to
   compare. The pick may be moved or replaced in the main window between
   tests; the view then resets to the new pick.

The display:

- By default, the clustered localizations are colored by cluster and the
  unclustered ones are hidden.
- ``Display non-clustered localizations`` shows all picked localizations in
  red, so the clustered ones appear white.
- ``Display cluster centers`` shows the cluster centers (for G5M, the
  molecules) over the clustered localizations. With both boxes ticked, the
  unclustered localizations are red, the clustered ones yellow and the
  centers blue.
- ``One pixel blur`` and the ``Contrast`` slider set the rendering. The
  contrast is kept between tests, so results can be compared at the same
  contrast.
- ``XY projection``, ``XZ projection`` and ``YZ projection`` switch the view
  of 3D data; ``Full FOV`` returns to the whole pick.
- :kbd:`Alt` (:kbd:`Option` on macOS) + :kbd:`W`/:kbd:`A`/:kbd:`S`/:kbd:`D`
  move the view and :kbd:`Alt` + :kbd:`=`/:kbd:`-` zoom in and out.

``Cluster entire dataset`` applies the current algorithm and parameters to
all localizations of a channel, or of every channel (the files are then
saved with a suffix you enter), and saves the clustered localizations. The
cluster centers are saved as well (``_centers.hdf5``) if
``Display cluster centers`` is ticked; G5M always saves them.

.. _render-nearest-neighbor-analysis:

Nearest Neighbor Analysis
-------------------------

Calculates distances to the ``k``-th nearest neighbors between two channels
(can be the same channel). ``k`` is defined by the user. The distances are
stored in nm as a .hdf5 localizations file with new columns ``nnd_1``,
``nnd_2``, ..., ``nnd_k`` for each localization in channel 1. The distances
are calculated in 3D if both datasets have z information.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``nn_analysis`` works on coordinate arrays in nm, so convert ``x`` and
   ``y``; ``z`` is in nm already.

   .. code-block:: python

      import numpy as np
      from picasso import io, postprocess

      locs1, info1 = io.load_locs("channel1_locs.hdf5")
      locs2, info2 = io.load_locs("channel2_locs.hdf5")

      X1 = locs1[["x", "y"]].to_numpy()                 # nm
      X2 = locs2[["x", "y"]].to_numpy()
      # 3D: X = locs[["x", "y", "z"]].to_numpy()

      nnd = postprocess.nn_analysis(X1, X2, nn_count=3)      # (N1, 3)
      for k in range(nnd.shape[1]):
          locs1[f"nnd_{k + 1}"] = nnd[:, k]
      io.save_locs("channel1_locs_nnd.hdf5", locs1, info1)

.. _render-cbc:

Coordinate-Based Colocalization (CBC)
-------------------------------------

Scores how strongly each localization of one channel colocalizes with
another channel (`Malkusch, et al., Histochem Cell
Biol, 2012 <https://doi.org/10.1007/s00418-011-0880-5>`__). For every
localization, the numbers of localizations of its own channel and of the
partner channel within radii up to ``Max. radius`` are normalized by the
area (2D) or volume (3D) searched and compared with the Spearman rank
correlation. The result is weighted by ``exp(-E / Max. radius)``, where ``E``
is the distance to the nearest partner localization. CBC values range from
-1 (anticorrelated) through 0 (uncorrelated) to 1 (colocalized).
Localizations without any partner localization, or without another
localization of their own channel, within ``Max. radius`` get the value set
in ``CBC without neighbors`` (0 by default), or are removed
if ``Remove localizations without neighbors`` is ticked. With several
partners, a localization is removed if it has no neighbors for at least one
of them.

The CBC is **not symmetric**: the values of channel A's localizations
against channel B differ from those of B's localizations against A. For
example, if B labels only half of A's structures, all of B is colocalized
with A, but only half of A is colocalized with B.

.. note::

   Link the localizations first (``Postprocess > Link localizations``, see
   :ref:`render-link-localizations`). Unlinked localizations count each binding event several times, which inflates the neighbor counts. Picasso warns about unlinked channels but lets you continue if requested.

To use CBC, load at least two channels and open
``Postprocess > Coordinate-based colocalization (CBC)...``:

1. Under ``Channel names``, give each channel a short name. It is used for
   the column names (``cbc_<name>``), so that long file names do not end up
   in them. The names are remembered while Render is open.
2. Choose the ``Scored channel``, whose localizations get the CBC values,
   and the ``Partner channel`` they are scored against.
3. ``All pairs of channels (both directions)`` scores every channel against
   every other channel instead, i.e., both directions of every pair. With
   two channels loaded, this gives both directions of the pair.
4. Set ``Max. radius (nm)``, the number of ``Radius steps`` up to it and the
   value given to localizations without neighbors (``CBC without
   neighbors``), or tick ``Remove localizations without neighbors`` to remove
   them instead. ``3D`` uses z if all channels involved contain it.
5. Click ``OK`` and choose where to save. A single pair is saved under the
   file name given; with all pairs, each channel is saved as
   ``<name>_cbc.hdf5`` in the chosen folder.

The saved localizations of the scored channel have a new column
``cbc_<name>`` per partner channel, named after the partner's name (e.g.,
``cbc_PSD95``). Characters other than letters, digits and underscores are
replaced by underscores (e.g., ``Cav2.1`` gives ``cbc_Cav2_1``). They can be filtered and rendered like any other
column, for example in Filter or by rendering by property (see
:ref:`render-properties`).

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``cbc`` works on localizations directly (``r_max`` in nm, as x, y and
   z) and returns a copy with the new column and the updated metadata.
   Several partners can be passed at once. Swap the channels to get the other
   direction, or use ``cbc_all_pairs`` to score every channel against every
   other one. ``undefined`` sets the value for localizations without
   neighbors; ``remove_undefined=True`` removes them instead.

   .. code-block:: python

      from picasso import io, postprocess

      locs_a, info_a = io.load_locs("channelA_locs.hdf5")
      locs_b, info_b = io.load_locs("channelB_locs.hdf5")

      locs_a, info_a = postprocess.cbc(
          locs_a, info_a, locs_b, partner_names="B", r_max=200, steps=100
      )  # new column "cbc_B"
      io.save_locs("channelA_locs_cbc.hdf5", locs_a, info_a)

      # the other direction differs
      locs_b, info_b = postprocess.cbc(locs_b, info_b, locs_a, partner_names="A")

      # all pairs of channels, both directions; channel i gets a column
      # "cbc_<name>" for every other channel
      locs_c, info_c = io.load_locs("channelC_locs.hdf5")
      all_locs, all_infos = postprocess.cbc_all_pairs(
          [locs_a, locs_b, locs_c],
          [info_a, info_b, info_c],
          names=["A", "B", "C"],
          undefined=0,
      )
      for locs, info, name in zip(all_locs, all_infos, "ABC"):
          io.save_locs(f"channel{name}_locs_cbc.hdf5", locs, info)

.. _render-ripley-h:

Ripley's H Function
-------------------

Shows whether the molecules of one channel are clustered, randomly
distributed or dispersed, and on which length scales. Ripley's K function
counts the neighbors within a distance ``r`` of each localization, normalized
by the density (`Ripley, J R Stat Soc B, 1977
<https://doi.org/10.1111/j.2517-6161.1977.tb01615.x>`__). Its transform, the H
function, ``H(r) = sqrt(K(r) / pi) - r``, is:

* 0 for complete spatial randomness (CSR),
* positive if the molecules are clustered at the scale ``r``,
* negative if they are dispersed (regularly spaced) at the scale ``r``.

By default, ``H`` is computed in 2D (x and y); it can also be computed in
3D, see :ref:`render-ripley-h-3d`.

.. note::

   Ripley's H describes how points are distributed, the case of SMLM this
   usually refers to *molecules*, so it should be computed on them, i.e., one localization per molecule, such as the
   molecules found by :ref:`G5M <render-g5m>` or the cluster centers saved by
   :ref:`DBSCAN <render-dbscan>`, :ref:`HDBSCAN <render-hdbscan>` or the
   :ref:`SMLM clusterer <render-smlm-clusterer>`. Otherwise, the repeated
   localizations of each molecule appear as clustering at the scale of the
   localization precision. Picasso recognizes molecules by their
   ``n_events`` or ``std_frame`` column and warns about channels without
   them, but lets you continue if requested.

``H`` is computed within regions of interest whose area gives the density, so
the regions should lie within the sample (e.g., inside the cell): the empty
parts of a region appear as clustering. Pairs near the edge of a region are
weighted by Ripley's isotropic edge correction, i.e., by the inverse of the
fraction of the circle around the localization that lies inside the region,
since the neighbors outside of it are not seen.

To use it, open ``Postprocess > Spatial statistics > Ripley's H function...``:

1. Choose the ``Channel`` (with several channels loaded). Leave ``Partner
   channel`` at ``None (one channel)``; choose a second channel for the cross
   function between two channels, see :ref:`render-cross-functions`.
2. Choose the ``Regions``: the ``Picks`` (any pick shape; ``H`` is computed
   in each pick and averaged over the picks with at least two localizations),
   the ``Current view`` or the ``Whole field of view``.
3. Set ``Max. radius (nm)``, which should be well below the size of the
   regions, and the number of ``Radius steps`` up to it.
4. Set the number of ``Simulations``. Each region is filled with as many
   uniformly random points as it holds localizations, and ``H`` is computed
   the same way; the lowest and highest simulated mean ``H`` at each radius
   give the CSR envelope. A mean ``H`` above the envelope indicates
   significant clustering, below it significant dispersion, at a pointwise
   significance level of ``2 / (simulations + 1)``, i.e., 0.02 for 99
   simulations. 0 skips the simulations.
5. ``Edge correction`` is on by default. Without it, ``H`` is biased
   downwards, increasingly with the radius.
6. Tick ``3D (x, y and z)`` to compute ``H`` in 3D (only for channels with
   z), see :ref:`render-ripley-h-3d`.
7. Click ``OK``.

The plot shows the mean ``H`` with the CSR envelope and, for several picks,
the function of each pick. ``Save data`` in the toolbar saves two .csv files:
the functions by radius (``r_nm``, ``h_mean_nm``, ``envelope_low_nm``,
``envelope_high_nm`` and ``h_region_<i>_nm`` for every region, numbered like
the picks) and, as ``<name>_regions.csv``, the number of localizations, area,
density, maximum of ``H`` and its radius of every region (in 3D also its z
range and volume, with the density per volume).

.. _render-ripley-h-3d:

3D
~~

In 3D, each region is the pick (or the view or field of view) extended over
a z range, the volume replaces the area and ``H(r) = (3 K(r) / (4 pi))^(1/3)
- r``. The edge correction weights each pair by the inverse of the fraction
of the sphere's surface around the localization that lies inside the region.
The dialog shows these settings when ``3D (x, y and z)`` is ticked:

- ``z range``: ``Localizations in each region`` extends each region from the
  lowest to the highest z of its localizations. ``Manual`` sets one range
  (``z min / max (nm)``) for all regions; localizations outside of it are
  ignored. The volume, and thus the density, depends on this range.
- ``Simulated z``: ``Uniform over the z range`` spreads the simulated points
  evenly in z, i.e., complete spatial randomness in 3D. Localizations are
  rarely uniform in z (the detection falls off away from focus, cells have a
  limited thickness), and such an uneven z distribution appears as
  clustering. ``Observed z values, shuffled`` gives the simulated points the
  z values of the localizations instead, which tests for clustering beyond
  the z distribution but also hides real clustering in z (e.g., a layer of
  molecules).

.. note::

   3D samples are usually thin (about 1 µm), so spheres larger than about a
   third of the z range mostly reach the top or bottom of the region and
   ``H`` relies on large edge correction weights, which makes it noisy.
   Picasso warns if ``Max. radius`` exceeds a third of the z range of any
   region. The poorer axial localization precision also stretches clusters in
   z, which raises ``H`` at short distances.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``ripley_h`` takes picks in nm with the pick shape and size as saved by
   Render (``io.load_picks``; the size of circular picks is the diameter) or,
   without picks, uses the whole field of view. It returns a dictionary with
   the radii, the function of each region, their mean and the CSR envelope,
   see :func:`picasso.postprocess.ripley_h`.

   .. code-block:: python

      from picasso import io, postprocess

      locs, info = io.load_locs("molecules.hdf5")
      picks, shape, size = io.load_picks("picks.yaml")

      result = postprocess.ripley_h(
          locs, info, picks, shape, size, r_max=500, steps=100, n_simulations=99
      )
      result["r"]  # radii (nm)
      result["h_mean"]  # mean H over the picks (nm)
      result["envelope_low"], result["envelope_high"]  # CSR envelope (nm)
      result["r_h_max"]  # radius of maximum H in each pick (nm)

      fig = postprocess.plot_ripley_h(result)
      fig.savefig("ripley_h.png")

      # 3D between z = -300 and 300 nm, simulated z shuffled from the data
      result_3d = postprocess.ripley_h(
          locs,
          info,
          picks,
          shape,
          size,
          r_max=200,
          use_z=True,
          z_range=(-300, 300),
          simulated_z="observed",
      )
      result_3d["volumes"]  # volume of each pick (nm^3)

.. _render-rdf:

Radial Distribution Function (RDF)
----------------------------------

The radial distribution function ``g(r)``, also called pair correlation
function, is the density of neighbors at the distance ``r`` from a molecule,
relative to the mean density. It is:

* 1 for complete spatial randomness (CSR),
* above 1 at the distances where molecules are more likely to be found than at
  random, e.g., within clusters or at a regular spacing between molecules,
* below 1 at the distances where they are depleted.

While Ripley's H function (see :ref:`render-ripley-h`) accumulates all
neighbors up to ``r``, ``g(r)`` counts the neighbors in a thin ring at ``r``
(in 3D, a spherical shell). It therefore shows at which distances molecules are
found, e.g., the spacing of a regular arrangement, rather than up to which
distance they are clustered. Each ring holds fewer pairs, so ``g(r)`` is
noisier than ``H``: use wider rings (fewer ``Radius steps``) for regions with
few molecules.

``g(r)`` is computed like Ripley's H function, with the same regions, edge
correction, 3D option (:ref:`render-ripley-h-3d`) and CSR envelope, and the
same note on molecules applies. Open ``Postprocess > Spatial statistics >
Radial distribution function (RDF)...``; the dialog is the same as for Ripley's
H function, with rings of 10 nm (50 ``Radius steps`` up to 500 nm) by default.

The plot shows the mean ``g(r)`` at the ring centers with the CSR envelope
and, for several picks, the function of each pick. ``Save data`` saves the
functions by radius (``r_nm``, ``g_mean``, ``envelope_low``,
``envelope_high`` and ``g_region_<i>``) and, as ``<name>_regions.csv``, the
number of localizations, area and density of every region (in 3D also its z
range and volume).

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``rdf`` takes the same arguments as ``ripley_h`` and returns the ring
   centers, the function of each region, their mean and the CSR envelope, see
   :func:`picasso.postprocess.rdf`.

   .. code-block:: python

      from picasso import io, postprocess

      locs, info = io.load_locs("molecules.hdf5")
      picks, shape, size = io.load_picks("picks.yaml")

      result = postprocess.rdf(
          locs, info, picks, shape, size, r_max=500, steps=50, n_simulations=99
      )
      result["r"]  # ring centers (nm)
      result["g_mean"]  # mean g(r) over the picks
      result["envelope_low"], result["envelope_high"]  # CSR envelope

      fig = postprocess.plot_rdf(result)
      fig.savefig("rdf.png")

.. _render-cross-functions:

Two Channels (Cross Functions)
------------------------------

Ripley's H function and the radial distribution function can also be computed
between two channels, e.g., two proteins, to show whether they are found
together (colocalization), independently of each other, or apart
(segregation), and on which length scales. Instead of the neighbors of the
same channel, the cross functions count the molecules of the partner channel
around each molecule of the first channel, relative to the density of the
partner (`Lotwick and Silverman, J R Stat Soc B, 1982
<https://doi.org/10.1111/j.2517-6161.1982.tb01221.x>`__):

* ``H > 0`` and ``g > 1``: the channels are found together at the scale
  ``r``,
* ``H = 0`` and ``g = 1``: the channels are independent,
* ``H < 0`` and ``g < 1``: the channels avoid each other.

The edge correction centers on the molecule of the first channel, so swapping
the channels gives slightly different counts near the edges; both directions
are combined into one symmetric function, so the order of the channels does
not matter.

.. note::

   Align the channels first (``Postprocess > Align channels``). An offset
   between the channels moves and blurs the peak of the cross functions at
   short distances.

To use it, choose a ``Partner channel`` in the dialog of either function. The
``Envelope`` sets how the two channels are simulated:

- ``Random shifts (independence)`` shifts the partner channel by a random
  offset, wrapping around the region (toroidal shift); in circular picks, it
  rotates it by a random angle about the center, and in 3D it also shifts z.
  Each channel keeps its own arrangement, e.g., its clusters, so the envelope
  is the range expected for two independent channels. This answers the usual
  question of colocalization: are the molecules of the two channels found
  together more often than by chance? The partner can only be wrapped around
  rectangular regions (box, square and rectangle picks, rectangular polygons,
  the current view and the field of view) and circular picks; for other
  polygon picks and brush picks, only random labels are available.
- ``Random labels (same distribution)`` pools the molecules of both channels
  and shuffles which channel each belongs to. It works in any region, but it
  answers a different question: do both channels follow the same
  distribution? Two channels that are clustered independently of each other
  fall below this envelope, since the pooled molecules are more clustered than
  the pairs of different channels, so do not read it as segregation.

The plot is titled *Ripley's cross H function* or *Cross radial distribution
function*. The saved summary of each region also lists the number and density
of the partner's molecules. In 3D, both channels need z; by default, each
region extends over the z range of the molecules of both channels.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   Pass the partner channel as ``partner_locs`` to ``ripley_h`` or ``rdf``;
   ``cross_null`` is ``"shift"`` (random shifts, default) or ``"labels"``
   (random labels).

   .. code-block:: python

      from picasso import io, postprocess

      locs_a, info_a = io.load_locs("protein_a_molecules.hdf5")
      locs_b, info_b = io.load_locs("protein_b_molecules.hdf5")
      picks, shape, size = io.load_picks("picks.yaml")

      result = postprocess.rdf(
          locs_a,
          info_a,
          picks,
          shape,
          size,
          partner_locs=locs_b,
          cross_null="shift",
          r_max=300,
          steps=30,
      )
      result["g_mean"]  # mean cross g(r) over the picks
      result["n_locs_partner"]  # molecules of the partner in each pick

      fig = postprocess.plot_rdf(result)
      fig.savefig("cross_rdf.png")

.. _render-nnd-distributions:

Nearest Neighbor Distances (All Channels)
-----------------------------------------

Plots the distributions of the first nearest neighbor distances (NNDs) of the
molecules within and between channels, together with the distributions
expected if the molecules do not interact, from random simulations. For every
ordered pair of channels (A, B), it histograms the distance from each molecule
of A to the nearest molecule of B in the same region; within a channel (A, A),
to the nearest other molecule. The plot is a grid with one panel per pair:
rows are the channel the distances start from, columns the channel of the
nearest neighbor. NNDs are not symmetric.

Unlike :ref:`render-nearest-neighbor-analysis`, which saves the distances of
each localization, this compares their distributions with random simulations.
As for Ripley's H function, it should be computed on molecules (see the note
in :ref:`render-ripley-h`), and the channels must be aligned.

Each region is simulated as many times as set in ``Simulations``:

- Within a channel (diagonal panels), its molecules are replaced by as many
  uniformly random points (complete spatial randomness).
- Between channels, as set in ``Envelope`` (see
  :ref:`render-cross-functions`): ``Random shifts (independence)`` shifts B
  at random, wrapping around the region, so that each channel keeps its own
  arrangement; ``Random labels (same distribution)`` shuffles the channels of
  the molecules of A and B.

The dashed line is the mean of the simulated histograms, the expected
distribution without interaction, and the shaded band is their range (lowest
to highest). The simulations are in the same regions as the data, so the
longer NNDs near the edges of a region, whose neighbors outside of it are not
seen, are expected as well; no edge correction is applied. NNDs shorter than
expected indicate clustering within a channel or colocalization between
channels, longer ones dispersion or segregation.

To use it, open ``Postprocess > Spatial statistics > Nearest neighbor
distances (all channels)...``:

1. Check the ``Channels`` to compare (with several channels loaded).
2. Choose the ``Regions`` (picks, current view or whole field of view).
3. Set ``Max. NND (nm)``, the largest distance shown, and the number of
   histogram ``Bins``. Longer NNDs are not shown but count in the fractions.
4. Set the number of ``Simulations`` and, with several channels, the
   ``Envelope``.
5. ``3D (x, y and z)`` computes the distances in 3D (all channels need z),
   with the same z settings as in :ref:`render-ripley-h-3d`; the simulated z
   applies within channels.
6. Click ``OK``.

``Save data`` saves the histograms (fraction of the molecules of A per bin;
columns ``<A>_to_<B>``, ``<A>_to_<B>_simulated_mean``,
``<A>_to_<B>_envelope_low`` and ``<A>_to_<B>_envelope_high``) and, as
``<name>_summary.csv``, the number of molecules and the median NND, measured
and simulated, of every pair.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``nnd_distributions`` takes the molecules of one or more channels and the
   regions like ``ripley_h``; the histograms are indexed by (from, to)
   channel, see :func:`picasso.postprocess.nnd_distributions`.

   .. code-block:: python

      from picasso import io, postprocess

      locs_a, info_a = io.load_locs("protein_a_molecules.hdf5")
      locs_b, info_b = io.load_locs("protein_b_molecules.hdf5")

      result = postprocess.nnd_distributions(
          [locs_a, locs_b],
          info_a,  # whole field of view
          names=["A", "B"],
          r_max=200,
          steps=40,
          n_simulations=99,
      )
      result["histograms"][0, 1]  # A -> B, fraction of A's molecules per bin
      result["medians"]  # median NND of each pair (nm)
      result["simulated_medians"]  # expected without interaction (nm)

      fig = postprocess.plot_nnd_distributions(result)
      fig.savefig("nnd.png")
