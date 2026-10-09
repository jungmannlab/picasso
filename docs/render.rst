Render
======

.. figure:: /images/render.png
   :width: 720px
   :alt: Picasso Render main window showing a rendered super-resolution image of localizations

``Picasso: Render`` displays the super-resolution image reconstructed from
localization files. It also provides a plethora of tools to process
localizations, for example: correct drift, pick and inspect regions of
interest, count binding sites with qPAINT, cluster localizations and map
molecules. This page
covers opening files and moving around the image; the cards below lead to the
individual topics and to a reference of every menu.

.. _render-opening-files:

Opening Files
-------------

1. In ``Picasso: Render``, open a
   movie file by dragging a localization file (ending with '.hdf5') into the
   window or by selecting ``File > Open``. Zoom into a region by dragging a rectangle with the left mouse button and pan by dragging with
   the right mouse button; see :ref:`render-navigation` for all mouse and
   keyboard controls.
2. You can adjust rendering options by selecting
   ``View > Display Settings`` (see :ref:`render-display-settings`):

   - The field 'Display pixel size (nm)' defines the size of the rendered
     pixels of the super-resolution image. This cannot go smaller that the current screen pixel size.
   - The contrast settings ``Min. Density`` and ``Max. Density`` define at
     which number of localizations per super-resolution pixel the minimum and
     maximum color of the colormap should be applied. They can be typed in or dragged on the two-handle slider below them.

3. (Optional) For multiplexed image acquisition, open HDF5 localization files
   from other channels subsequently. Alternatively, drag and drop all HDF5 files to be displayed simultaneously.

.. _render-coloring:

How localizations are colored
-----------------------------

The colors of the image depend on what is loaded:

- **One file:** the density of localizations is shown with the colormap set in
  ``View > Display settings`` (see :ref:`render-colormap-setting`).
- **One file with a** ``group`` **column** (e.g., after picking or
  clustering): each group is drawn in one of eight colors, chosen by the group
  id (``group`` modulo 8), so neighboring groups are told apart. The colormap
  is not used.
- **Several files:** each file (channel) is drawn in its own color or
  colormap, set in ``View > Files...`` (see :ref:`render-files`). The
  ``group`` column is then ignored for coloring.
- **Render by property** (one file only) overrides the above and colors the
  localizations by a column of your choice, e.g., ``z`` or ``frame`` (see
  :ref:`render-properties`).

.. _render-navigation:

Navigating the image
--------------------

What the mouse buttons do on the image depends on the active
tool, selected in the ``Tools`` menu:

- :ref:`Zoom <render-zoom>` (:kbd:`Ctrl+Z`, the default): drag with the left
  button to zoom into a rectangle, drag with the right button to pan.
- :ref:`Pick <render-pick>` (:kbd:`Ctrl+P`) selects regions of localizations
  for further analysis (see :doc:`render/picking`): the left button draws a
  pick, the right button removes the pick under the cursor (for polygon and
  brush picks, it undoes the last vertex or stroke instead).
- :ref:`Measure <render-measure>` (:kbd:`Ctrl+M`) measures distances on the
  image: the left button adds a point, the right button finishes the current
  measurement or, clicked again, deletes the last one.
- :ref:`Move <render-move>` (:kbd:`Ctrl+G`): drag with the left button to move
  the localizations of the selected channels, e.g., to register channels by
  eye. The right button has no function here.

The pick shape and size, the channels moved and the appearance of the tools
are set in ``Tools > Tools settings...`` (:kbd:`Ctrl+T`); see
:ref:`render-tools-settings`.

The following controls move the view; those marked *every tool* also work
while picking or measuring, so the tool need not be changed to look around.

- **Zoom**:

  - With the Zoom tool, drag a rectangle with the left mouse button to zoom
    to it. The rectangle stretches towards the bottom right; releasing above
    or left of the start cancels.
  - :kbd:`Shift` + the left button drags the same rectangle in *every tool*.
  - :kbd:`Ctrl` (:kbd:`Cmd` on macOS) + the mouse wheel (or trackpad scroll)
    and a trackpad pinch zoom about the cursor; :kbd:`Ctrl` +/- zoom about
    the center.

- **Pan**: drag with the right mouse button (Zoom tool), or, in *every tool*,
  with the middle mouse button, with :kbd:`Ctrl` (:kbd:`Cmd`) + the left
  button or with :kbd:`Alt` (:kbd:`Option` on macOS) + the left button. The
  arrow keys (or :kbd:`W`/:kbd:`A`/:kbd:`S`/:kbd:`D`) move the view by a
  fraction of the window.
- **Fit**: ``View > Fit image to window`` (:kbd:`Ctrl+W` or :kbd:`Home`)
  shows the whole image; a triple click with the Zoom tool does the same.

The shortcuts for moving and zooming with the keyboard work anywhere in the window (the arrow keys or
:kbd:`W`/:kbd:`A`/:kbd:`S`/:kbd:`D`, :kbd:`Ctrl` +/-).

The 3D view uses the same controls, plus rotation; see
:ref:`render-3d-navigation`.

.. _render-qpaint:

qPAINT
------

qPAINT counts the binding sites in a pick from the frequency of its binding
events (`Jungmann et al., Nature Methods, 2016 <https://doi.org/10.1038/nmeth.3804>`__). In Render:

1. Pick structures with a known number of binding sites and calibrate the
   influx rate in ``View > Show info`` (see :ref:`render-info-picks`).
2. Pick the structures to count. The Info dialog shows the number of units as
   mean and standard deviation over the picks, and
   ``File > Save pick properties`` saves it for every pick (see
   :ref:`render-save-pick-properties`).

.. _render-topics:

Topics
------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: :octicon:`circle;1.5em;sd-mr-1` Picking regions of interest
      :link: render/picking
      :link-type: doc

      The six pick shapes, picking, Pick similar and saving picks.

   .. grid-item-card:: :octicon:`gear;1.5em;sd-mr-1` Tools
      :link: render/tools
      :link-type: doc

      The zoom, pick, measure and move tools, their settings and appearance.

   .. grid-item-card:: :octicon:`sliders;1.5em;sd-mr-1` Display settings and info
      :link: render/display-settings
      :link-type: doc

      Contrast, colormaps, blur methods, scale bar, render by property and the
      info dialog.

   .. grid-item-card:: :octicon:`git-compare;1.5em;sd-mr-1` Drift correction
      :link: render/drift
      :link-type: doc

      AIM, marker-based and RCC drift correction, and how to inspect, undo or
      reapply drift.

   .. grid-item-card:: :octicon:`sync;1.5em;sd-mr-1` 3D view
      :link: render/3d
      :link-type: doc

      Rotating 3D data, navigation in 3D and building animations.

   .. grid-item-card:: :octicon:`graph;1.5em;sd-mr-1` Clustering and molecular mapping
      :link: render/analysis
      :link-type: doc

      RESI, G5M molecular mapping, clustering and nearest neighbor analysis.

   .. grid-item-card:: :octicon:`file;1.5em;sd-mr-1` File menu
      :link: render/menu-file
      :link-type: doc

      Opening, saving and exporting localizations and pick regions.

   .. grid-item-card:: :octicon:`eye;1.5em;sd-mr-1` View menu
      :link: render/menu-view
      :link-type: doc

      Datasets and colors, overlay images, slicing and linked windows.

   .. grid-item-card:: :octicon:`tools;1.5em;sd-mr-1` Tools menu
      :link: render/menu-tools
      :link-type: doc

      Pick similar, pick fiducials, traces, masks and the other pick
      utilities.

   .. grid-item-card:: :octicon:`workflow;1.5em;sd-mr-1` Postprocess menu
      :link: render/menu-postprocess
      :link-type: doc

      Drift, grouping, linking, alignment, expressions and clustering.

   .. grid-item-card:: :octicon:`cpu;1.5em;sd-mr-1` Performance
      :link: render/performance
      :link-type: doc

      CPU usage on shared workstations and GPU rendering.

.. toctree::
   :hidden:

   render/picking
   render/tools
   render/display-settings
   render/drift
   render/3d
   render/analysis
   render/menu-file
   render/menu-view
   render/menu-tools
   render/menu-postprocess
   render/performance
