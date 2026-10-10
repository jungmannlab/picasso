.. _render-dialogs:

Display Settings and Info
=========================

This page describes two dialogs of the ``View`` menu:

- :ref:`Display settings <render-display-settings>` (:kbd:`Ctrl+D`) control
  how the localizations are rendered: zoom, contrast and colormap, blur
  method, scale bar and rendering by property.
- :ref:`Info <render-show-info>` (:kbd:`Ctrl+I`) reports on the loaded data:
  the current field of view, the localization precision and NeNA, the FRC
  resolution and statistics of the picks.

.. _render-display-settings:

Display Settings
----------------

Open via ``View > Display settings...`` (:kbd:`Ctrl+D`). The dialog sets how
the image is rendered and applies every change immediately, to all channels. Linked windows can share
these settings, see :ref:`render-link-settings`. The 3D view has its own dialog, see :ref:`render-3d-display-settings`.

.. _render-display-general:

General
~~~~~~~

Adjust the general display settings.

Zoom
   The number of screen pixels per camera pixel. For example, at ``Zoom`` 10
   with a camera pixel size of 130 nm, one screen pixel shows 13 nm. The value
   follows every zoom in the window; typing in a new one zooms to it.
Display pixel size (nm)
   Set the size of the pixel in the rendered image. Choose ``dynamic`` to
   automatically adjust to current window size when zooming.
Minimap
   Click ``show minimap`` to display a minimap in the upper left corner to
   see where the current field of view is within the whole image.

.. _render-colormap-setting:

Contrast
~~~~~~~~

``Min. Density`` and ``Max. Density`` set the number of localizations per
display pixel that are mapped onto the lowest and the highest color of the
colormap; they can also be dragged on the slider below them. ``Colormap``
offers over 100 colormaps. The last option, ``Custom``, loads a ``.npy`` file
with a numpy array of shape (256, 4) and values between 0 and 1.

The selected colormap will be saved when closing render, under ``Colormap`` in
the ``Render`` section of ``~/.picasso/settings.yaml`` (see
:ref:`user-settings-file`), and restored the next time Render starts.

.. note::

   This colormap is used only when a single localization file is loaded
   (without a ``group`` column and without rendering by property, see
   :ref:`render-coloring`). With
   several files, each channel is drawn in its own color or colormap, chosen
   in ``View > Files...`` (see :ref:`render-files`; custom colormaps for
   channels are built there, see :ref:`render-custom-colormaps`). The
   colormap here is then not used.

.. _render-blur:

Blur
~~~~

Select a blur method. ``Min. blur (nm)`` sets the smallest Gaussian width the
precision-based methods draw. Available options are:

None
   Each localization adds one count to the display pixel it falls in (a
   histogram).
One-Pixel-Blur
   The histogram blurred with a Gaussian of :math:`\sigma = 1` display pixel
   in x and y.
Global Localization Precision
   Every localization is drawn with the same Gaussian :math:`\sigma` equal to the median precision of
   the whole channel.
Individual Localization Precision
   Each localization is drawn as a Gaussian whose width is its own
   localization precision (``lpx``, ``lpy``).
   Note that this blurs the data a second time by the localization error
   already contained in the positions, which costs a factor of about 1.4 in
   resolution (`Baddeley, Cannell and Soeller, Microscopy and Microanalysis, 2010 <https://doi.org/10.1017/S143192760999122X>`__).
Individual Localization Precision, iso
   As above, with a round Gaussian whose width is the mean of ``lpx`` and
   ``lpy``.
Adaptive Histogram (Quad-Tree)
   The quad-tree adaptive histogram
   (`Baddeley, Cannell and Soeller, Microscopy and Microanalysis, 2010 <https://doi.org/10.1017/S143192760999122X>`__). A bin is split into four while it holds more localizations than the *leaf
   capacity*, so every bin has about the same signal-to-noise ratio whatever
   the local density:

   - Bin counts are Poisson distributed and bins hold between about a quarter
     of the capacity and the capacity, so the mean SNR is the square root of
     half the capacity (the paper's estimate; the dialog shows it).
   - The bin size shows the local sampling: large bins where localizations are
     sparse, small ones where they are dense.
   - Choose the capacity below the number of localizations of the smallest
     structure you want to see; structures with fewer than about half the
     capacity are merged into their surroundings, which suppresses spurious
     detail in undersampled regions.
   - The default of 10 (SNR about 2.2) suits DNA-PAINT data with tens of
     localizations per binding site; the original paper used 5 for STORM data.
   - A bin is never smaller than one display pixel. When zoomed out, a
     display pixel often holds more localizations than the leaf capacity, so
     most bins are single pixels and the image approaches the plain histogram
     (blur ``None`` above).
   - The mode renders on the CPU from the spatial index of each channel (see
     :ref:`files <spatial-index>`), which makes it fast at any zoom.
   - In the 3D view the same method is applied to the projected
     localizations: the tree is rebuilt from the rows in view for every
     orientation (a sort of those rows per frame, so whole large channels
     rotate more slowly than with the other methods; drag previews use a
     subset as usual).
Jittered Triangulation
   The adaptively jittered, averaged Delaunay triangulation
   (`Baddeley, Cannell and Soeller, Microscopy and Microanalysis, 2010 <https://doi.org/10.1017/S143192760999122X>`__). The localizations in view are triangulated and
   every triangle is drawn with an intensity inverse to its area, which is
   linear in the local density:

   - To blur to the local sampling limit, every localization is displaced by a
     random jitter whose width is its mean distance to its neighbors (times the
     **Jitter** factor: 1, the paper's choice, or 0.5 for known periodic
     structures).
   - The images of **Passes** many such triangulations are averaged (25 by
     default).
   - Dense regions therefore keep their resolution while isolated
     localizations are dimmed instead of shown as confident dots, and there is
     no second blur by the localization precision.
   - It is computed on the CPU (the passes in parallel) and quite
     computationally expensive, so it is a mode for zoomed-in views: above
     **Max. localizations in view** (100,000 by default) the histogram is
     rendered instead, and a note below the triangulation settings says how many localizations are in view.
   - While panning and zooming a single unjittered triangulation is previewed
     and the average follows when the mouse pauses.
   - In the 3D view the projected localizations are triangulated for every
     orientation. The limit there is ``Max. localizations`` in the 3D view's
     display settings (copied from the main window when the 3D view opens),
     and it counts all localizations loaded into the 3D view, not only those
     in view (see :ref:`render-3d-display-settings`).

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   The blur methods are the ``blur_method`` of ``render.render``;
   ``min_blur_width`` is in camera pixels.

   .. code-block:: python

      from picasso import io, render

      locs, info = io.load_locs("movie_locs.hdf5")
      n_locs, image = render.render(
          locs, info, disp_px_size=10,
          blur_method="gaussian",       # None, "gaussian", "gaussian_iso", "smooth", "convolve"
          min_blur_width=0.0,
      )


.. _render-camera:

Camera
~~~~~~

``Camera pixel size (nm)`` is read from the metadata (``.yaml`` file) of the
localizations, or set to a default value if it is missing. It converts camera
pixels to nm, e.g., for the scale bar and the ``Display pixel size``.

.. _render-scale-bar:

Scale bar
~~~~~~~~~

Tick ``Scale bar`` to show a scale bar of ``Scale bar length (nm)``, based on
the ``Camera pixel size``. ``Automatic length`` keeps it at roughly 1/8 of the
window width as you zoom, and ``Print scale bar length`` prints the length
next to it.

.. _render-properties:

Render properties
~~~~~~~~~~~~~~~~~

Colors the localizations by one of their columns (only for single-channel
data): choose the ``Parameter``, its ``Min.`` and ``Max.``, the number of
``Colors`` and the ``Colormap``, and tick ``Render``. The colormap chosen here is kept
separately from the channel colormap above, under ``Colormap Property`` in the
``Render`` section of ``~/.picasso/settings.yaml`` (default ``gist_rainbow``,
see :ref:`user-settings-file`), and restored the next time a property is
rendered.

.. _render-colorbar-format:

Color bar (LUT)
^^^^^^^^^^^^^^^

When rendering by property is active, exporting an image additionally saves
the color bar (LUT) next to it, named after the image with the suffix
``_colorbar`` (e.g., ``locs_view.png`` and ``locs_view_colorbar.png``) - for
instance, to annotate z color-coding in a figure. It displays the colors that
localizations are rendered with, one band per color as set by ``Colors``, and
the property values along the bar.

This applies to ``Export current view``, ``Export complete image`` and
``Export view manually``, as well as to ``Export current view`` in the
3D view. The property, its limits, the number of colors and the
colormap are additionally written to the ``.yaml`` file that accompanies the
exported image.

The color bar is saved as a ``.png`` by default. To save it as a vector
graphic instead - whose bands, ticks and text stay editable in figure
software - set ``Colorbar format`` to ``.svg`` under ``Render`` in
``~/.picasso/settings.yaml`` (also available under File > Picasso settings in
any module):

.. code-block:: yaml

    Render:
      Colorbar format: .svg

The setting applies to the color bar only; the image itself keeps the format
chosen in the save dialog.

.. _render-show-info:

Show Info
---------

Open via ``View > Show info...`` (:kbd:`Ctrl+I`). The dialog reports on the
loaded data, the current view, pick properties and qPAINT.

.. _render-info-display:

Display
~~~~~~~

``Image width`` and ``Image height``
   The size of the image area of the window, in screen pixels.
``View X / Y`` and ``View width / height``
   The top-left corner and the size of the current field of view, in camera
   pixels.
``Renderer``
   Whether the image is rendered on the GPU (with the GPU's name) or by CPU
   threads.
``Change field of view``
   Opens a dialog to type in the field of view (``X``, ``Y``, ``Width``,
   ``Height`` in camera pixels).
``Save FOV`` and ``Load FOV``
   Save the current field of view to a ``.txt`` file and restore it later,
   e.g., to show the same region of several datasets. A saved ``.txt`` file
   can also be dropped onto the main window.

.. _render-info-precision:

Precision
~~~~~~~~~

.. _render-nena:

``Median localization precision`` shows the median of the first channel.
``Calculate NeNA`` estimates the average localization precision via the NeNA
(nearest neighbor based analysis) approach, and ``Show NeNA plot`` displays
its fit (`Endesfelder et al., Histochemistry and Cell Biology, 2014
<https://doi.org/10.1007/s00418-014-1192-3>`__). The NeNA value is added to
the metadata of the channel, so it is saved with the localizations.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   NeNA is returned in camera pixels.

   .. code-block:: python

      from picasso import io, lib, postprocess

      locs, info = io.load_locs("movie_locs.hdf5")
      pixelsize = lib.get_from_metadata(info, "Pixelsize")

      result, nena_px = postprocess.nena(locs, info)
      print(f"NeNA: {nena_px * pixelsize:.1f} nm")
      fig = postprocess.plot_nena(result)


.. _render-info-frc:

FRC
~~~

Estimates the resolution by Fourier ring correlation (`Nieuwenhuizen et al.,
Nature Methods, 2013 <https://doi.org/10.1038/nmeth.2448>`__):

- ``Calculate FRC`` splits the localizations of the **current field of view**
  (cropped to a square) randomly into two halves, renders an image of each
  with a pixel size set by NeNA, and correlates them. ``FRC resolution (nm)``
  is where the FRC curve drops below 1/7. Zoom to a region of interest first;
  a large field of view takes long.
- ``Save rendered images`` also saves the two images as ``_1.tif`` and
  ``_2.tif``.
- ``Show FRC plot`` shows the FRC curve.
- The Q factor correction for repeated localizations of the same molecule
  (blinking, DNA-PAINT rebinding) is not applied, so the value reported may be
  optimistic.

``FRC in several ROIs`` (collapsed by default) estimates the uncertainty of
the FRC resolution:

- ``Calculate FRC in ROIs`` places up to ``Number of ROIs`` (30)
  non-overlapping random square ROIs of ``ROI side length (µm)`` (5 µm)
  across the whole image, not only the current field of view. ROIs with fewer
  than ``Min. localizations per ROI`` (1,000) are not used.
- The FRC is calculated in each ROI with the same NeNA-based pixel size, and
  ``FRC resolution, ROIs (nm)`` shows their mean ± standard deviation.
- ``Review ROIs`` opens a window listing the ROIs with their positions,
  localization counts and resolutions, an overview of where they lie (used:
  green, excluded: gray, selected: red) and the FRC curve of the selected
  ROI. Untick an ROI, e.g., one on a cell edge or background, to exclude it
  from the mean.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``frc`` works on one viewport (``((y_min, x_min), (y_max, x_max))`` in
   camera pixels), ``frc_rois`` on random ROIs over the whole image.

   .. code-block:: python

      frc = postprocess.frc(locs, info, viewport=((0, 0), (64, 64)))
      print(frc["resolution"])                        # nm
      fig = postprocess.plot_frc(frc)

      rois = postprocess.frc_rois(
          locs, info, viewport=((0, 0), (512, 512)),
          n_rois=30, roi_size=5000.0, min_locs=1000,  # roi_size in nm
      )
      print(rois["resolutions"].mean(), rois["resolutions"].std())


.. _render-info-fov:

Field of view
~~~~~~~~~~~~~

``No. of localizations in FOV`` shows the number of localizations in the
current field of view.

.. _render-info-picks:

Picks, kinetics and qPAINT
~~~~~~~~~~~~~~~~~~~~~~~~~~

Statistics of the picks and qPAINT (counting binding sites from the binding
kinetics, `Jungmann et al., Nature Methods, 2016
<https://doi.org/10.1038/nmeth.3804>`__). ``Number of picks`` is updated as
you pick. Press ``Calculate info below`` to fill in the mean and standard
deviation over the picks of:

- ``No. of localizations`` and ``No. of events`` (binding events) per pick;
- ``RMSD to COM (nm)``, the root mean square distance of the localizations to
  the pick's center of mass, and ``RMSD in z (nm)`` for 3D data;
- ``Bright time (frames)`` and ``Dark time (frames)``, the mean bright and
  dark times of each pick, fitted to their cumulative distributions.

``Ignore dark times <=`` sets the longest gap (in frames) within one binding
event: localizations separated by at most this many frames are linked into
one event. It is used for all the values above and for the pick properties
saved from the ``File`` menu.

qPAINT:

1. Pick structures with a known number of binding sites, enter it in
   ``No. of units per pick``, press ``Calculate info below`` and then
   ``Calibrate influx``. The ``Influx rate (1/frames)`` (the qPAINT index of
   one binding site) is computed from the pooled dark times:
   :math:`1 / (\bar{\tau}_d \cdot n_\mathrm{units})`. It can also be typed
   in.
2. Pick the structures to count and press ``Calculate info below``.
   ``Number of units`` shows, for each pick, :math:`1 / (\text{influx rate}
   \cdot \tau_d)` with its dark time :math:`\tau_d`, as mean and standard
   deviation over the picks.

``Show fitted binding kinetics`` plots the distributions of the bright and dark
times with their fits.

Dark times are counted as the number of frames without signal between two
binding events in a pick; see "HDF5 Pick Property Files" in the file format
documentation (:doc:`/files`) for the exact convention.

.. important::

   Since Picasso 0.11.3, dark times are one frame shorter than in earlier
   versions, so influx rates calibrated with earlier versions should be
   recalibrated.
