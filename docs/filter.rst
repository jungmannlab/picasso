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
   (:kbd:`Ctrl+D`). The axes of lengths (``x``, ``sx``, ``lpx``, ...) are
   labeled with ``(nm)``.
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

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   The table is a ``pandas.DataFrame``, so a filter is boolean indexing.
   ``lib.calculate_optimal_bins`` gives the bins Filter uses for its
   histograms. Recording each range as ``"column": [min, max]`` in a
   metadata entry made by ``lib.step_info("Filter", ...)``, as Filter
   records it, makes the steps reapplicable, see :ref:`filter-metadata`.

   .. code-block:: python

      import matplotlib.pyplot as plt
      from picasso import io, lib

      locs, info = io.load_locs("movie_locs.hdf5")

      plt.hist(locs["sx"], bins=lib.calculate_optimal_bins(locs["sx"]))

      # Keep sx between 100 and 135 nm and at least 500 photons
      ranges = {"sx": [100.0, 135.0], "photons": [500.0, float("inf")]}
      for column, (min_value, max_value) in ranges.items():
          locs = locs[locs[column].between(min_value, max_value)]

      # Record the ranges as Filter does, so that they can be reapplied
      info = lib.append_metadata(info, lib.step_info("Filter", **ranges))
      io.save_locs("movie_locs_filter.hdf5", locs, info)


Numerical filtering
~~~~~~~~~~~~~~~~~~~

In the menu bar, click ``Filter > Filter numerically...`` (:kbd:`Ctrl+F`). A dialog is displayed where the user can numerically filter values for any of the columns. Click the ``Filter`` button in the dialog to remove localizations which do not fit in the input parameters.

The limits are inclusive: localizations with values equal to ``min`` or ``max`` are kept. Localizations with non-finite values (NaN, infinity) in the column are always removed.

.. _filter-expression:

Filtering by expression
~~~~~~~~~~~~~~~~~~~~~~~

Click ``Filter > Filter by expression...`` (:kbd:`Ctrl+E`) to keep the
localizations that satisfy a condition on their columns, for example:

.. code-block:: text

   photons > 500 and 100 < sx < 160
   lpx < 10
   photons > median(photons)
   sqrt(lpx**2 + lpy**2) < 2 * mean(sqrt(lpx**2 + lpy**2))

Type the condition and press :kbd:`Enter` or click ``Filter``. The dialog
lists the names the condition can use and reports typos while typing; after
filtering, it shows how many localizations were kept.

The condition is written like a Python expression:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Element
     - Meaning
   * - Column names
     - The values of each localization, e.g., ``x``, ``photons``, ``lpx``.
       Every length (``x``, ``y``, ``z``, ``sx``, ``lpx``, ...) is in nm, so
       ``lpx < 10`` keeps localizations more precise than 10 nm.
   * - ``pixelsize``
     - The camera pixel size in nm, from the metadata. Rarely needed now
       that the columns are in nm; e.g., ``x < 10 * pixelsize`` selects the
       first ten camera pixels.
   * - ``pi``, ``inf``, ``nan``
     - Constants.
   * - ``< <= > >= == !=``
     - Comparisons, also chained: ``100 < sx < 160``.
   * - ``and``, ``or``, ``not``
     - Combine conditions. ``&``, ``|`` and ``~`` work too, but each
       comparison then needs parentheses: ``(photons > 500) & (sx < 160)``.
   * - ``+ - * / // % **``
     - Arithmetic; ``**`` is the power (``^`` is not supported).
   * - ``abs``, ``sqrt``, ``log``, ``log10``, ``exp``, ``floor``, ``ceil``,
       ``isfinite``
     - Functions applied to each localization.
   * - ``mean``, ``median``, ``std``, ``min``, ``max``, ``sum``,
       ``percentile(values, q)``
     - Statistics over the localizations currently shown (``q`` from 0 to
       100), ignoring NaN and infinite values. Their argument can be any
       expression, e.g., ``median(sqrt(lpx**2 + lpy**2))``.

Comparisons with NaN are false, so ``photons > 500`` also removes
localizations whose ``photons`` is NaN.

Statistics are computed over the localizations left by the previous filters,
so the order of the filters matters: ``photons > median(photons)`` keeps half
of the localizations that are shown when it is applied.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``expression.filter_locs`` applies one or more conditions, in order, to a
   ``DataFrame``; ``pixelsize`` (if a condition uses it) is taken from the
   metadata. The returned metadata records the conditions as Filter does,
   so that they can be reapplied (see :ref:`filter-metadata`).
   ``expression.condition`` returns the boolean mask instead.

   .. code-block:: python

      from picasso import io, expression

      locs, info = io.load_locs("movie_locs.hdf5")
      locs, info = expression.filter_locs(
          locs,
          info,
          conditions=["photons > median(photons)", "lpx < 10"],
      )
      io.save_locs("movie_locs_filter.hdf5", locs, info)

   The same is available in the :ref:`command line <cmd-filter>`.

.. _filter-metadata:

Filters from metadata
~~~~~~~~~~~~~~~~~~~~~

The filtering information is stored in the metadata .yaml file.
``Picasso: Filter`` allows the user to load the filtering steps from previously
filtered data by clicking ``Filter > Apply filters from metadata...`` or by
dragging the .yaml file into the main window (with localizations already
open). The extracted information is displayed to the user before approval.

All filter steps found in the metadata are applied in the order they were
applied originally, also over several Filter sessions: histogram selections,
numerical filters and :ref:`expressions <filter-expression>`. The order
matters for expressions with statistics such as ``photons > median(photons)``.
The removed columns of every step are removed at the end. Steps on columns
that the current localizations do not have are listed and skipped.

The metadata of each Filter session lists its steps as conditions under
``Filter steps``; histogram and numerical filters appear as ranges such as
``100 < sx < 160``. Each column's limits are also stored as
``column: [min, max]``, which is all that files saved before Picasso 1.0
contain; such files are replayed from these limits.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``lib.apply_filter_steps`` replays the steps recorded in the metadata of
   a filtered file (``recipe``) on other localizations. Besides the filtered
   localizations and their metadata, which records the applied steps, it
   returns the columns the recipe names but the data lacks.
   ``lib.extract_filter_recipe`` returns the steps without applying them.

   .. code-block:: python

      from picasso import io, lib

      locs, info = io.load_locs("movie_locs.hdf5")
      recipe = io.load_info("other_locs_filter.yaml")
      locs, info, missing = lib.apply_filter_steps(locs, info, recipe=recipe)
      io.save_locs("movie_locs_filter.hdf5", locs, info)


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
:ref:`render-g5m-overfitting`.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``one_sided=True`` tests only for fewer events in the clustered
   population.

   .. code-block:: python

      from picasso import io, clusterer, lib

      mols, info = io.load_locs("molecules.hdf5")
      clustered, sparse = clusterer.test_subclustering(
          mols, info, clustering_dist=25, sparse_dist=80
      )
      lib.plot_subclustering_check(
          clustered, sparse, "subclustering.png",
          clustering_dist=25, sparse_dist=80,
          one_sided=False,
      )
