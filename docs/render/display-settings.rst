.. _render-dialogs:

Display Settings and Info
=========================

Two dialogs control how the image is displayed and report on the loaded data:
the Display Settings dialog and the info dialog.

.. _render-display-settings:

Display Settings
----------------

Allows to change the display settings. Open via ``View > Display Settings``
(:kbd:`Ctrl+D`).

.. _render-display-general:

General
~~~~~~~

Adjust the general display settings.

Zoom
   Set the magnification factor.
Display pixel size (nm)
   Set the size of the pixel in the rendered image. Choose ``dynamic`` to
   automatically adjust to current window size when zooming.
Minimap
   Click ``show minimap`` to display a minimap in the upper left corner to
   localize where the current field of view is within the image.

.. _render-colormap-setting:

Contrast
~~~~~~~~

Define the minimum and maximum density of the and select a colormap. Over 100
colormaps are available. The last option ``Custom`` requires the user to load
their own ``.npy`` file containing a numpy array with a custom colormap.

The selected colormap will be saved when closing render, under ``Colormap`` in
the ``Render`` section of ``~/.picasso/settings.yaml`` (see
:ref:`user-settings-file`), and restored the next time Render starts.

Colormaps built with the custom colormap editor (``Edit custom colormaps`` in
the **Datasets** dialog, see :ref:`render-custom-colormaps`, one per channel
with its own list of color stops) are kept separately, under
``Render: CustomColormaps``, keyed by name.

.. _render-blur:

Blur
~~~~

Select a blur method. Available options are:

None
   Each localization adds one count to the display pixel it falls in (a
   histogram).
One-Pixel-Blur
   The histogram blurred with a Gaussian of one display pixel.
Global Localization Precision
   Every localization is drawn with the same Gaussian, the median precision of
   the whole channel (computed once per channel).
Individual Localization Precision
   Each localization is drawn as a Gaussian whose width is its own
   localization precision (``lpx``, ``lpy``); *iso* uses the mean of the two.
   Note that this blurs the data a second time by the localization error
   already contained in the positions, which costs a factor of about 1.4 in
   resolution (Baddeley, Cannell & Soeller, *Microsc. Microanal.* 2010).
Adaptive Histogram (Quad-Tree)
   The quad-tree adaptive histogram of Baddeley, Cannell & Soeller (2010). A
   bin is split into four while it holds more localizations than the **leaf
   capacity**, so every bin has about the same signal-to-noise ratio whatever
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
   - Bins never split below a display pixel, so zoomed out the image is the
     histogram.
   - The mode renders on the CPU from the spatial index of each channel (see
     :ref:`files <spatial-index>`), which makes it fast at any zoom.
   - In the 3D rotation window the same method is applied to the projected
     localizations: the tree is rebuilt from the rows in view for every
     orientation (a sort of those rows per frame, so whole large channels
     rotate more slowly than with the other methods; drag previews use a
     subset as usual).
Jittered Triangulation
   The adaptively jittered, averaged Delaunay triangulation of Baddeley,
   Cannell & Soeller (2010). The localizations in view are triangulated and
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
     rendered instead and the dialog says so.
   - While panning and zooming a single unjittered triangulation is previewed
     and the average follows when the mouse pauses.
   - In the 3D rotation window the projected localizations are triangulated
     for every orientation, within the same limit.

.. tip::

   In scripts, select the jittered triangulation with
   ``blur_method="triangulation"`` together with ``triangulation_passes`` and
   ``triangulation_jitter`` in ``picasso.render.render`` and
   ``render_scene``; ``picasso.render.triangulation.render_triangulation``
   works on arrays.

.. _render-camera:

Camera
~~~~~~

Select the pixel size of the camera. This will be automatically set to a
default value or the value specified in the ``*.yaml`` file.

.. _render-scale-bar:

Scale Bar
~~~~~~~~~

Activate scale bar. The length of the scale bar is calculated with the Pixel
Size set in the Camera dialog. Activate ``Print scale bar length`` to
additionally print the length.

.. _render-properties:

Render properties
~~~~~~~~~~~~~~~~~

This allows rendering properties by color. The colormap chosen here is kept
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
``Export view manually``, as well as to ``Export current view`` in the 3D
rotation window. The property, its limits, the number of colors and the
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

Displays the info dialog. Open via ``View > Show info``.

.. _render-info-display:

Display
~~~~~~~

Shows the image width/height, the coordinates, and dimensions of the current
FoV.

.. _render-info-movie:

Movie
~~~~~

Displays the median fit precision of the dataset. Clicking on ``Calculate``
allows calculating the precision via the NeNA approach. See
`DOI: 10.1007/s00418-014-1192-3 <https://doi.org/10.1007/s00418-014-1192-3>`_.

.. _render-info-frc:

FRC
~~~

Displays the FRC resolution of the dataset. Takes in the image in the current
FOV and calculates the FRC resolution via splitting the localizations into two
halves. Based on the approach from
`10.1038/nmeth.2448 <https://doi.org/10.1038/nmeth.2448>`_. Does not take into
account the Q factor for multiple blinking.

.. _render-info-fov:

Field of view
~~~~~~~~~~~~~

Shows the number of localizations in the current FoV.

.. _render-info-picks:

Picks
~~~~~

Allows calculating statistics about the picked localizations. Press
``Calculate info below`` to calculate.

- ``Ignore dark times`` allows treating consecutive localizations as on, even
  if there are localizations (specified by the parameter) missing between
  them.
- When defining the number of units per pick, you can calibrate the influx
  rate via ``Calibrate influx``.
- A histogram of the dark and bright time can be plotted when clicking
  ``Histograms``.

Dark times are counted as the number of frames without signal between two
binding events in a pick; see "HDF5 Pick Property Files" in the file format
documentation (:doc:`/files`) for the exact convention.

.. important::

   Since Picasso 0.11.3, dark times are one frame shorter than in earlier
   versions, so influx rates calibrated with earlier versions should be
   recalibrated.
