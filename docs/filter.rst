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

This function can be applied to molecular maps/cluster centers which save the
column ``n_events``, i.e., the number of binding events detected per molecule.

The premise is the following: a single molecule is expected to give rise to a
certain distribution of the number of binding events. If extra molecules are
assigned, the number of binding events per molecule will on average be lower
than the distribution would predict.

Thus, by comparing the distribution of
the number of binding events per molecule for two populations (clustered vs.
sparse), one can assess whether subclustering has occurred.

To plot the two distributions, use ``Plot > Test subclustering...``. The dialog
allows the user to set:

- the maximum nearest neighbors distance between molecules to be considered as
  clustered (``Max. dist. between clustered molecules (nm)``);
- the minimum nearest neighbor distance for sparse molecules
  (``Min. dist. between sparse molecules (nm)``).

The numbers of events for the two populations can be saved to a CSV file by
checking the ``Save histogram values`` checkbox before clicking the
``Test subclustering`` button.
