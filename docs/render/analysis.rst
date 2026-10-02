.. _render-analysis:

Analysis
========

``Picasso: Render`` includes RESI and G5M for resolving and mapping individual
molecules, several clustering algorithms and a nearest neighbor analysis. All
of them are found in the ``Postprocess`` menu (see :doc:`menu-postprocess`).

.. _render-resi:

RESI
----

.. figure:: /images/render_resi.png
   :width: 374px
   :alt: Picasso Render RESI dialog with per-channel clustering parameters

   The RESI dialog.

In Picasso 0.6.0, a new RESI (Resolution Enhancement by Sequential Imaging)
dialog was introduced. It allows for a substantial resolution boost by
sequential imaging of a single target with multiple labels with Exchange-PAINT
(*Reinhardt, Masullo, Baudrexel, Steen, et al., Nature, 2023.* DOI:
10.1038/s41586-023-05925-9).

To use RESI:

1. Prepare your individual RESI channels (localization, undrifting, filtering
   and **alignment**).
2. Load such localization lists into Picasso Render and open
   ``Postprocess > RESI``. The dialog shown above will appear.
3. Each channel will be clustered using the SMLM clusterer (other clustering
   algorithms could be applied as well although only the SMLM clusterer is
   implemented for RESI in Picasso). Clustering parameters can be defined for
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
   channels to create the final RESI file.

.. _render-g5m:

G5M
---

In Picasso 0.9.5, a new algorithm for molecular mapping (i.e., finding the
positions of individual molecules from localizations) was introduced: G5M
(Gaussian Mixture Modeling with Modifications for Molecular Mapping;
Kowalewski, Reinhardt et al. *Nature Comms*, 2026. DOI:
10.1038/s41467-026-70198-5). It is opened via
``Postprocess > Molecular mapping (G5M)``.

G5M is based on Gaussian Mixture Modeling (GMM) but includes several
modifications to make it suitable for molecular mapping. All the
technicalities as well as the user guide of the method are explained in the
publication mentioned and its Supplementary Information. Please refer to
``picasso.g5m`` for the details of the implementation. Below is a brief
summary of the user guide.

.. _render-g5m-preprocessing:

Preprocessing
~~~~~~~~~~~~~

G5M requires some preprocessing of localizations to filter out the badly
fitted ones, especially the ones arising from crosstalk (overlapping
blinking):

- These can be excluded from 2D data where the ellipticity and size of the
  image of an emitter in x and y can be filtered (in Picasso these are found
  under names "ellipticity", "sx" and "sy", respectively).
- Moreover, the photon count can be cut-off as crosstalk is likely to result
  in a higher-intensity signal.
- In 3D data these filters are less reliable due to astigmatism, however,
  "d_zcalib" could be used.

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
:ref:`render-aim`). This prevents overfitting of too many molecules.

.. _render-g5m-clustering:

Clustering before G5M
~~~~~~~~~~~~~~~~~~~~~

Prior to molecular mapping, clustering of localizations is required to split
the data into smaller chunks. For many datasets, :ref:`DBSCAN <render-dbscan>`
works well. While in some cases some adjustments may be needed, we recommend
the following DBSCAN parameters:

- In 2D, DBSCAN radius (epsilon) of 2*LP, in 3D - 3*LP (LP - average
  localization precision of the dataset, for example, NeNA or median
  localization precision).
- Default min. samples is set to 4.

Clustering in Picasso adds the ``group`` column to the localization file,
which is required for G5M.

.. important::

   G5M relies on the information in the ``group`` column, therefore, if it is
   overwritten (for example, by picking localizations after DBSCAN
   clustering), G5M will not work.

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
events (saved in the column ``n_events``). In the publication, we recommend a
threshold of at least 3 binding events per molecule.

The final postprocessing step is log-likelihood filtering (using the column
``p_val``). The recommended threshold is ``> 0.0015``, however, it might need
to be adjusted for your data, especially in 3D this can be too conservative.

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
  well-behaved data.
- **Relative σ values** (as of v0.9.8), i.e., the fitted Gaussian σ divided by
  the average loc. precision around the molecule. This can be used to estimate
  if the loc. precision values are accurate (if not, many molecules will have
  relative σ values close to the user-selected min./max. σ).
- **KS 2 sample test** (as of version 0.9.10) comparing the two
  distributions. The output test statistic and theoretical p value correspond
  to the KS test, while the permutation p value is calculated by randomly
  permuting the labels of clustered and sparse molecules 1,000 times and
  calculating the fraction of permutations that result in a KS test statistic
  as extreme as the one observed with the original labels.

.. _render-g5m-troubleshooting:

Troubleshooting
~~~~~~~~~~~~~~~

If the outcome of G5M seems unsatisfactory, please check the following:

- Make sure that ``group`` column is present in the localization file and
  contains the correct information (i.e., from DBSCAN clustering, not from
  picking localizations). ``group_input`` can also be used.
- Make sure that the loc. precision values (columns ``lpx``, ``lpy``,
  ``lpz``) are correct, comparing NeNA and median loc. precision is a
  reasonable proxy (without fiducial markers); the most common issue is a
  miscalibrated camera, leading to incorrect photon counts and thus incorrect
  loc. precisions.
- Consider using a more accurate fitting model, such as spline fitting; we
  found that switching to it with optional increase in max. sigma can be very
  beneficial.
- Another reason why the loc. precision values can be off is due to the small
  box size in the localization step; especially in 3D astigmatic imaging,
  single-emitter images can be quite large, potentially exceeding the
  user-defined box size; in such cases, we recommend increasing the box size
  in the localization step and rerunning the analysis.
- Inspect if the localizations were preprocessed as described above.
- Rerun the analysis without postprocessing (filtering) and redo it manually,
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
  G5M takes too long to run, the DBSCAN clusters most likely contain too many
  molecules. In such a case, we recommend splitting such clusters further.

.. _render-clustering:

Clustering
----------

The clustering algorithms are found in the ``Postprocess > Clustering``
submenu.

.. tip::

   It is highly recommended to remove any fiducial markers before clustering
   (with any of the algorithms), to lower clustering time, given they are of
   no interest to the user. To do that, the markers can be picked and removed
   using ``Tools > Remove localizations in picks``.

.. _render-dbscan:

DBSCAN
~~~~~~

Cluster localizations with the dbscan clustering algorithm.

.. _render-hdbscan:

HDBSCAN
~~~~~~~

Cluster localizations with the hdbscan clustering algorithm.

.. _render-smlm-clusterer:

SMLM clusterer
~~~~~~~~~~~~~~

Cluster localizations with the custom algorithm designed for SMLM. In short,
localizations with the maximum number of neighboring localizations within a
user-defined radius are chosen as cluster centers, around which all
localizations within the given radius belong to one cluster. If two or more
local maxima are within the radius, the clusters are merged.

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

.. _render-test-clusterer:

Test clusterer
~~~~~~~~~~~~~~

Opens a dialog where different clustering parameters can be checked on the
loaded dataset. Requires a single pick region of interest to be selected.

.. _render-nearest-neighbor-analysis:

Nearest Neighbor Analysis
-------------------------

Calculates distances to the ``k``-th nearest neighbors between two channels
(can be the same channel). ``k`` is defined by the user. The distances are
stored in nm as a .hdf5 localizations file with new columns ``nnd_1``,
``nnd_2``, ..., ``nnd_k`` for each localization in channel 1. The distances
are calculated in 3D if both datasets have z information.
