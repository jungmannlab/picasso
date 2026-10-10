.. _render-analysis:

Clustering and Molecular Mapping
================================

This page describes RESI and G5M for resolving and mapping individual molecules, several clustering algorithms and a nearest neighbor analysis. All
of them are found in the ``Postprocess`` menu (see :doc:`menu-postprocess`).

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

   The channels must be aligned already. Radii are in camera pixels and may
   be given per channel as lists.

   .. code-block:: python

      from picasso import io, lib, postprocess

      paths = ["round1_locs.hdf5", "round2_locs.hdf5"]
      locs, infos = zip(*[io.load_locs(p) for p in paths])
      pixelsize = lib.get_from_metadata(infos[0], "Pixelsize")

      resi_locs, resi_info = postprocess.resi(
          list(locs), list(infos),
          radius_xy=10 / pixelsize,       # e.g., 2 x NeNA
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

      from picasso import clusterer, g5m, io, lib

      locs, info = io.load_locs("movie_locs.hdf5")
      pixelsize = lib.get_from_metadata(info, "Pixelsize")

      clustered, _ = clusterer.dbscan(
          locs, radius=20 / pixelsize, min_samples=4,
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

   Radii are in camera pixels; ``pixelsize`` is only needed for 3D data.

   .. code-block:: python

      from picasso import clusterer, io, lib

      locs, info = io.load_locs("movie_locs.hdf5")
      pixelsize = lib.get_from_metadata(info, "Pixelsize")

      clustered, cluster_info = clusterer.dbscan(
          locs, radius=20 / pixelsize, min_samples=10, min_locs=10, pixelsize=pixelsize
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
          locs, min_cluster_size=10, min_samples=10, pixelsize=pixelsize
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

   ``radius_xy`` and ``radius_z`` are in camera pixels (z needs
   ``pixelsize``); ``frame_analysis`` is the basic frame analysis.
   ``find_cluster_centers`` gives one localization per cluster.

   .. code-block:: python

      clustered, cluster_info = clusterer.cluster(
          locs, radius_xy=20 / pixelsize, min_locs=10, frame_analysis=True,
          radius_z=None, pixelsize=pixelsize, progress="console",
      )
      centers = clusterer.find_cluster_centers(clustered, pixelsize, progress="console")
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
      from picasso import io, lib, postprocess

      locs1, info1 = io.load_locs("channel1_locs.hdf5")
      locs2, info2 = io.load_locs("channel2_locs.hdf5")
      pixelsize = lib.get_from_metadata(info1, "Pixelsize")

      X1 = locs1[["x", "y"]].to_numpy() * pixelsize
      X2 = locs2[["x", "y"]].to_numpy() * pixelsize
      # 3D: X = np.column_stack([locs[["x", "y"]].to_numpy() * pixelsize, locs["z"]])

      nnd = postprocess.nn_analysis(X1, X2, nn_count=3)      # (N1, 3)
      for k in range(nnd.shape[1]):
          locs1[f"nnd_{k + 1}"] = nnd[:, k]
      io.save_locs("channel1_locs_nnd.hdf5", locs1, info1)
