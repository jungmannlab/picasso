Filter
======

``Picasso: Filter`` shows the localizations of an HDF5 file as a table and
removes localizations by their properties, either graphically in 1D and 2D
histograms or numerically. It also offers a test for subclustering in
molecular maps.

.. figure:: /images/filter.png
   :width: 720px
   :class: screenshot
   :alt: Picasso Filter window with the localization table, one row per localization and one column per property

   The localization table in ``Picasso: Filter``.

Filtering of localizations
--------------------------

1. Open a localization HDF5 file in ``Picasso: Filter`` by dragging it into the
   main window or by selecting ``File > Open``. The displayed table shows the
   properties of each localization in rows. Each column represents one property
   (e.g., coordinates, number of photons).
2. To display a histogram from values of one property, select the respective
   column in the header and select ``Plot > Histogram`` (:kbd:`Ctrl+H`). 2D
   histograms can be displayed by selecting two columns (press :kbd:`Ctrl` to
   select multiple columns) and then selecting ``Plot > 2D Histogram``
   (:kbd:`Ctrl+D`).
3. Left-click and hold the mouse button down to drag a selection area in a 1D
   or 2D histogram. The selected area will be shaded (in orange by default).
   Each localization event with histogram properties outside the selected area
   is immediately removed from the localization list.
4. Save the filtered localization table by selecting ``File > Save``.

The look of the histograms is set in ``Plot > Plot settings...``, or with the
*Plot settings* button in the toolbar of each histogram window: for example
the font sizes, the colors, the gridlines, etc. Changes are remembered across
Picasso apps.

The columns are explained in :doc:`files` for
:ref:`localizations <files-localization-hdf5>`,
:ref:`molecular maps <files-molecular-maps>` and
:ref:`pick properties <files-pick-properties>`.

The directory of the last file opened or saved is remembered across sessions,
under ``PWD`` in the ``Filter`` section of ``~/.picasso/settings.yaml`` (see
the :ref:`user settings file <user-settings-file>`).

Numerical filtering
~~~~~~~~~~~~~~~~~~~

In the menu bar, click ``Filter > Filter numerically...`` (:kbd:`Ctrl+F`). A dialog is displayed where the user can numerically filter values for any of the columns. Click the ``Filter`` button in the dialog to remove localizations which do not fit in the input parameters.

The limits are inclusive: localizations with values equal to ``min`` or ``max`` are kept. Localizations with non-finite values (NaN, infinity) in the column are always removed.

Filters from metadata
~~~~~~~~~~~~~~~~~~~~~

The filtering information is stored in the metadata .yaml file.
``Picasso: Filter`` allows the user to load the filtering steps from previously
filtered data by clicking ``Filter > Apply filters from metadata...``. The
extracted information is displayed to the user before approval.

All filter steps found in the metadata are applied: if
the file was filtered several times (in several Filter sessions), the ranges of
each column are intersected, so the strictest limits of all steps apply, and
the removed columns of every step are removed. Steps on columns that the
current localizations do not have are listed and skipped.

.. _filter-test-subclustering:

Test subclustering
~~~~~~~~~~~~~~~~~~

This test checks molecular maps, e.g., from G5M (see :ref:`render-g5m`), for
subclustering, i.e., for single molecules that were falsely split into several. It
needs the column ``n_events``, the number of binding events assigned to each
molecule. It was introduced in `Kowalewski, Reinhardt, et al., Nature
Communications, 2026 <https://doi.org/10.1038/s41467-026-70198-5>`__.

The premise is the following: a single molecule gives rise to a certain
distribution of the number of binding events. If it is split into several
molecules, its binding events are shared between them, so each gets fewer.
Split molecules lie close to each other, so subclustering shows up as
molecules with close neighbors having fewer binding events than isolated
ones.

The test compares two populations:

- **Clustered** molecules, whose nearest neighbor is closer than
  ``Max. dist. between clustered molecules (nm)`` (default 25 nm).
- **Sparse** molecules, whose nearest neighbor is at least
  ``Min. dist. between sparse molecules (nm)`` away (default 80 nm). This
  distance must be larger than the clustered one.

Molecules in between belong to neither population and are left out. The
distance is to the first nearest neighbor, in 3D if the molecules have a ``z``
column.

To run the test, use ``Plot > Test subclustering...``, set the two distances
and click ``Test subclustering``. To also save the numbers of events of both
populations, check ``Save histogram values`` first: they are saved as the
columns ``clustered_nevents`` and ``sparse_nevents`` of a CSV file
(``<file>_subcluster_test.csv`` by default), the shorter column padded with
empty values.

The plot shows a histogram of the number of events for each population, its
mean as a dashed line and the mean ± standard deviation in the legend. The
x axis spans the 2.5th to the 97.5th percentile of all values. The title
reports a two-sample Kolmogorov-Smirnov (KS) test between the two
populations:

``stat``
   The KS statistic.
``permutation p_value``
   The clustered and sparse molecules are pooled and their labels shuffled
   1,000 times, keeping the sizes of the two populations; no molecules are
   drawn or left out. The p value is the fraction of shuffles whose KS
   statistic is at least as large as the observed one, counting the observed
   arrangement as one of them (`Phipson and Smyth, 2010
   <https://doi.org/10.2202/1544-6115.1585>`__), i.e.,
   (count + 1) / 1,001. The smallest possible value is therefore about 0.001.
   The shuffles use a fixed random seed, so the same data always give the
   same p value.
``theoretical p_value``
   The p value of the KS test, from ``scipy.stats.ks_2samp``.

The test is two-sided: it detects any difference between the two
distributions, also if the clustered molecules have *more* events. Check in
the plot that a significant result comes from the clustered population being
shifted towards fewer events.

.. figure:: /images/filter-subclustering.png
   :width: 100%
   :class: only-light
   :alt: Two subclustering test plots; left, a well-behaved dataset whose clustered and sparse molecules have the same distribution of binding events, p value 0.93; right, a subclustered dataset whose clustered molecules have fewer binding events, p value 0.001

   **Left:** a well-behaved molecular map. Clustered and sparse molecules
   have the same distribution of binding events (means 12.8 and 13.0,
   permutation p value 0.93). **Right:** a subclustered molecular map. The
   clustered molecules have fewer binding events than the sparse ones (means
   10.3 and 13.2, permutation p value 0.001, the smallest possible).

.. figure:: /images/filter-subclustering-dark.png
   :width: 100%
   :class: only-dark
   :alt: Two subclustering test plots; left, a well-behaved dataset whose clustered and sparse molecules have the same distribution of binding events, p value 0.93; right, a subclustered dataset whose clustered molecules have fewer binding events, p value 0.001

   **Left:** a well-behaved molecular map. Clustered and sparse molecules
   have the same distribution of binding events (means 12.8 and 13.0,
   permutation p value 0.93). **Right:** a subclustered molecular map. The
   clustered molecules have fewer binding events than the sparse ones (means
   10.3 and 13.2, permutation p value 0.001, the smallest possible).

The same test runs automatically after G5M, see
:ref:`render-g5m-overfitting`. In Python:

.. code-block:: python

   from picasso import io, clusterer, lib

   mols, info = io.load_locs("molecules.hdf5")
   clustered, sparse = clusterer.test_subclustering(
       mols, info, clustering_dist=25, sparse_dist=80
   )
   lib.plot_subclustering_check(
       clustered, sparse, "subclustering.png",
       clustering_dist=25, sparse_dist=80,
       one_sided=False,  # True: test only for fewer events when clustered
   )
