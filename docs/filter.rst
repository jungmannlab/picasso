filter
=======

.. image:: ../docs/filter.png
   :scale: 50 %
   :alt: UML Design

Filtering of localizations
--------------------------
Open a localization HDF5 file in ``Picasso: Filter`` by dragging it into the main window or by selecting ``File`` > ``Open``. The displayed table shows the properties of each localization in rows. Each column represents one property (e.g., coordinates, number of photons); see the filetypes section for details.

To display a histogram from values of one property, select the respective column in the header and select ``Plot`` > 'Histogram' (Ctrl + h). 2D histograms can be displayed by selecting two columns (press Ctrl to select multiple columns) and then selecting ``Plot`` > ``2D Histogram`` (Ctrl + d).

Left-click and hold the mouse button down to drag a selection area in a 1D or 2D histogram. The selected area will be shaded in green. Each localization event with histogram properties outside the selected area is immediately removed from the localization list.

Save the filtered localization table by selecting ``File`` > ``Save``.