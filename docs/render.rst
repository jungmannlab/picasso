Render
======

.. figure:: /images/render.png
   :width: 720px
   :alt: Picasso Render main window showing a rendered super-resolution image of localizations

``Picasso: Render`` displays the super-resolution image reconstructed from
localization files. It also provides the tools to correct drift, pick and
inspect regions of interest, view 3D data, cluster localizations and map
molecules. This page covers opening files and moving around the image; the
cards below lead to the individual topics and to a reference of every menu.

.. _render-opening-files:

Opening Files
-------------

1. Rendering of the super-resolution image: In ``Picasso: Render``, open a
   movie file by dragging a localization file (ending with '.hdf5') into the
   window or by selecting ``File > Open``. The super-resolution image will be
   rendered automatically. Zoom into a region by dragging a rectangle with the
   left mouse button and pan by dragging with the right mouse button; see
   :ref:`render-navigation` for all mouse and keyboard controls.
2. (Optional) Adjust rendering options by selecting
   ``View > Display Settings`` (see :ref:`render-display-settings`):

   - The field 'Display pixel size (nm)' defines the size of the rendered
     pixels of the super-resolution image.
   - The contrast settings ``Min. Density`` and ``Max. Density`` define at
     which number of localizations per super-resolution pixel the minimum and
     maximum color of the colormap should be applied.
   - They can be typed in or dragged on the two-handle slider below them,
     whose track is logarithmic and spans the densities present in the
     rendered image.

3. (Optional) For multiplexed image acquisition, open HDF5 localization files
   from other channels subsequently. Alternatively, drag and drop all HDF5
   files to be displayed simultaneously.

.. _render-navigation:

Navigating the image
--------------------

The ``Tools`` menu selects the active tool (Zoom, Pick, Measure or Move;
:kbd:`Ctrl+Z`, :kbd:`Ctrl+P`, :kbd:`Ctrl+M`, :kbd:`Ctrl+G`). The following
controls move the view; those marked *every tool* also work while picking or
measuring, so the tool need not be changed to look around.

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

Moving and zooming with the keyboard (the arrow keys or
:kbd:`W`/:kbd:`A`/:kbd:`S`/:kbd:`D`, :kbd:`Ctrl` +/-) has no menu entries; the
shortcuts work anywhere in the window.

The 3D rotation window uses the same controls, plus rotation; see
:ref:`render-3d-navigation`.

.. _render-topics:

Topics
------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: :octicon:`git-compare;1.5em;sd-mr-1` Drift correction
      :link: render/drift
      :link-type: doc

      AIM, marker-based and RCC drift correction, and how to inspect, undo or
      reapply drift.

   .. grid-item-card:: :octicon:`circle;1.5em;sd-mr-1` Picking regions of interest
      :link: render/picking
      :link-type: doc

      The six pick shapes, picking, Pick similar and saving picks.

   .. grid-item-card:: :octicon:`sync;1.5em;sd-mr-1` 3D rotation window
      :link: render/3d
      :link-type: doc

      Rotating 3D data, navigation in 3D and building animations.

   .. grid-item-card:: :octicon:`graph;1.5em;sd-mr-1` Analysis
      :link: render/analysis
      :link-type: doc

      RESI, G5M molecular mapping, clustering and nearest neighbor analysis.

   .. grid-item-card:: :octicon:`sliders;1.5em;sd-mr-1` Display settings and info
      :link: render/display-settings
      :link-type: doc

      Contrast, colormaps, blur methods, scale bar, render by property and the
      info dialog.

   .. grid-item-card:: :octicon:`cpu;1.5em;sd-mr-1` Performance
      :link: render/performance
      :link-type: doc

      CPU usage on shared workstations and GPU rendering.

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

      Zoom, pick, measure and move tools and the pick utilities.

   .. grid-item-card:: :octicon:`workflow;1.5em;sd-mr-1` Postprocess menu
      :link: render/menu-postprocess
      :link-type: doc

      Drift, grouping, linking, alignment, expressions and clustering.

.. toctree::
   :hidden:

   render/drift
   render/picking
   render/3d
   render/analysis
   render/display-settings
   render/performance
   render/menu-file
   render/menu-view
   render/menu-tools
   render/menu-postprocess
