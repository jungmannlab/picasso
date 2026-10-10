First Steps
===========

This page walks you through a first DNA-PAINT analysis, from a raw movie to a
drift-corrected super-resolution image with your structures of interest
picked. It covers only the settings you need to get started; every step ends
with a link to the full documentation of that module for when you want more.

Picasso is a set of modules, each opening in its own window. Start them from
their shortcuts or from a terminal with ``picasso <module>`` (for example
``picasso localize``), depending on how you installed Picasso (see
:doc:`installation`).

.. tip::

   We provide an example dataset showing 20 nm grid DNA origami. It's a cropped movie from `Strauss and Jungmann. Nature Methods, 2020 <https://doi.org/10.1038/s41592-020-0869-x>`_ (Fig 2b, R6). You can practice all the steps listed below on this experimental dataset. Download the file from this `link <website>`_. Use ``EM Gain = 1``, ``Sensitivity = 0.53``, ``Baseline = 100`` and ``Pixel size = 130``.

.. rubric:: Key words used on this page

Frame
   One camera image of the movie. A DNA-PAINT movie has thousands of them.
Camera pixel
   A pixel of the raw movie, typically 100-160 nm wide in the sample. Positions
   in Picasso's files are given in camera pixels.
Spot
   The blurred image of a single fluorescent molecule on the camera, a few
   pixels wide. Its shape is stems from the point spread function (PSF).
Localization
   The position of one spot in one frame, measured with a precision far below
   the pixel size. The super-resolution image is built from millions of them.
Localization precision
   How well a single localization is determined, in camera pixels (``lpx``,
   ``lpy``). Brighter spots give better precision.

The steps
---------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: :octicon:`pencil;1.2em;sd-mr-1` 1 · Design
      :link: first-steps-design
      :link-type: ref
      :class-card: sd-card-hover

      *Optional.* Plan a DNA origami.

   .. grid-item-card:: :octicon:`play;1.2em;sd-mr-1` 2 · Simulate
      :link: first-steps-simulate
      :link-type: ref
      :class-card: sd-card-hover

      *Optional.* Test imaging conditions.

   .. grid-item-card:: :octicon:`location;1.2em;sd-mr-1` 3 · Localize
      :link: first-steps-localize
      :link-type: ref
      :class-card: sd-card-hover

      Find and fit the single molecules.

   .. grid-item-card:: :octicon:`filter;1.2em;sd-mr-1` 4 · Filter
      :link: first-steps-filter
      :link-type: ref
      :class-card: sd-card-hover

      *Optional.* Remove bad localizations.

   .. grid-item-card:: :octicon:`image;1.2em;sd-mr-1` 5 · Render
      :link: first-steps-render
      :link-type: ref
      :class-card: sd-card-hover

      Correct drift and pick structures.

   .. grid-item-card:: :octicon:`arrow-right;1.2em;sd-mr-1` 6 · Next
      :link: first-steps-next
      :link-type: ref
      :class-card: sd-card-hover

      Where to go from here.

.. _first-steps-design:

1 · Plan the experiment (optional)
----------------------------------

If you image DNA origami, :doc:`/design` lets you lay out DNA-PAINT docking
sites on a rectangular origami and gives you the staple sequences and
pipetting schemes to make it. Its designs can also be loaded into Simulate.

.. _first-steps-simulate:

2 · Test imaging conditions (optional)
--------------------------------------

:doc:`/simulate` generates a DNA-PAINT movie of known structures under the
imaging conditions you choose (imager concentration, laser power, number of
frames, ...), so you can check whether your structure can be resolved before
the real measurement. The simulated ``.raw`` movie is analyzed like a
measured one, starting with Localize below; note that Simulate adds no sample drift.

.. _first-steps-localize:

3 · Localize
------------

Localization turns the raw movie into a list of localizations around molecules' positions.

How localization works
~~~~~~~~~~~~~~~~~~~~~~

Picasso localizes in two steps:

**Identification**
   In every frame, Picasso looks for the bright spots of single molecules and
   places a square box around each one. At this point, the spot's position is
   only known to the nearest pixel.

**Fitting**
   A 2D Gaussian (a bell-shaped surface, the model of the spot) is fitted to
   the pixels inside each box. This gives the position of the molecule with a
   precision of a few nanometers, together with its number of photons, the
   background, the width of the spot and the localization precision.

.. figure:: /images/first-steps-localization.png
   :width: 100%
   :class: dark-light
   :alt: Left, one frame of a DNA-PAINT movie with a yellow box around each identified spot, as drawn in Localize; right, the 7 by 7 pixels of one box with the dashed outline of the fitted Gaussian and a green cross at the fitted position, which lies between pixels

   **Left: identification.** Each spot found in the frame gets a yellow box,
   as in Localize. **Right: fitting.** One box enlarged: a 2D Gaussian
   (dashed green outlines) is fitted to its pixels, and its center, the green
   cross, is the position of the molecule.

Open the movie
~~~~~~~~~~~~~~

1. Start ``Picasso: Localize`` and drag the movie into the window, or select
   ``File > Open movie...``.

   - Most microscope formats (``.tif``, ``.nd2``, ``.czi``, ...) open directly;
     see :ref:`localize-file-formats`.
   - A ``.raw`` movie asks for its size and data type, unless it comes with
     a ``.yaml`` file of the same name (as movies from Simulate do).

2. Browse the frames with the left and right arrow keys.
3. Drag the two-handle slider at the bottom of the window to adjust the contrast
   until the spots are clearly visible against the background.

Set the identification parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open ``Analyze > Parameters...``. Two settings in the ``Identification`` group
decide which spots are fitted.

**Box side length**
   The size, in camera pixels, of the box cut out around each spot. Only the
   pixels inside the box are fitted, so:

   - the box must contain the whole spot, otherwise the fit misjudges its
     position, brightness and width;
   - a box much larger than the spot picks up neighboring molecules and
     background, and spots too close together are no longer separated.

   A good value is 6σ + 1 rounded to an odd number, where σ is the width (the
   standard deviation) of the spot in camera pixels. On a well-aligned
   microscope σ is about 0.9 pixels, which gives the default of **7**.

**Min. net gradient**
   A score for how strongly a spot stands out from its background: it is
   larger for brighter and sharper spots. Only spots scoring above this
   threshold are fitted. The score depends on your experimental setup, so there
   is no universal value; find it by eye:

   1. Tick ``Preview``. The spots that would be identified with the current
      settings are now marked with boxes in the displayed frame.
   2. Lower the value until all spots you can see are boxed. If boxes appear
      on the background where there is no spot, the value is too low; if
      clear spots are left without a box, it is too high.
   3. Check a few frames with the arrow keys before moving on.

.. figure:: /images/first-steps-net-gradient.png
   :width: 100%
   :alt: Two Localize windows showing the same frame with Preview on; left, 32 red boxes, many of them on empty background; right, 14 red boxes, each on a spot

   The same frame with ``Preview`` on. **Left:** ``Min. net gradient`` is too
   low, and many boxes sit on background noise (32 spots found). **Right:** a
   good value (here: 5,000), with a box on every spot (14 spots found). Spots at the very
   edge of the frame get no box, because the box would not fit into the frame.

Set the camera parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

The camera records counts, not photons. The ``Photon Conversion`` group in the
``Parameters`` dialog converts one into the other, which is needed for correct photon numbers and
localization precisions. Take the values from your camera's data sheet or
acquisition software:

``EM Gain``
   The electron multiplication gain of an EMCCD camera; 1 for an sCMOS camera
   or with EM gain switched off.
``Baseline``
   The average count of a pixel that receives no light.
``Sensitivity``
   Electrons per count.
``Pixel size (nm)``
   The size of a camera pixel in the sample, i.e., the physical pixel size
   divided by the magnification of the microscope. It is saved with the
   localizations, and the later steps use it to convert camera pixels to
   nanometers, e.g., for drift correction, pick sizes and scale bars.

For movies made with Simulate, use ``EM Gain`` = 1, ``Baseline`` = 0,
``Sensitivity`` = 1 and the ``Pixel size`` set in Simulate (130 nm by
default).

.. tip::

   To avoid typing the values every time, store them for your camera in a
   camera config file; see :ref:`localize-camera-config`.

Fit and save
~~~~~~~~~~~~

1. In the ``Fit Settings`` group, keep the model ``2D elliptical Gaussian``
   and the default optimizer. If a ``Use GPU`` checkbox is shown, ticking it
   makes the fit faster.
2. Select ``Analyze > Localize (Identify & Fit)``. The progress is shown in
   the status bar at the bottom of the window.

When the fit is done, two files are saved next to the movie:

``<movie>_locs.hdf5``
   The localizations, one row per localization. This is the file you open in
   the next steps.
``<movie>_locs.yaml``
   The movie information and all settings used, readable in any text editor.
   It travels with the ``.hdf5`` file through the following steps, each adding
   its own settings. Note that the same information is contained in the ``.hdf5``
   file as well, so it is not strictly necessary to keep this file.

The columns you will meet most often:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``frame``
     - The frame the molecule was found in.
   * - ``x``, ``y``
     - The position, in camera pixels.
   * - ``photons``
     - The number of photons in the spot.
   * - ``bg``
     - The background, in photons per pixel.
   * - ``sx``, ``sy``
     - The width of the spot in x and y, in camera pixels.
   * - ``lpx``, ``lpy``
     - The localization precision in x and y, in camera pixels.
   * - ``net_gradient``
     - The identification score from above.

All columns are described in :ref:`files-localization-hdf5`.

3D localization with astigmatism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Skip this part for 2D data.*

For 3D imaging, a cylindrical lens is placed in the detection path. It makes
spots above the focus stretch in one direction and spots below the focus in
the other, so the shape of a spot tells its height ``z`` (`Huang et al.,
Science, 2008 <https://doi.org/10.1126/science.1153529>`__).
Picasso learns this relation once from a bead sample (the calibration) and
then uses it for every measurement.

**Calibrate (once per microscope setup)**

1. Record a z-stack of fluorescent beads on a coverslip: move the stage in
   steps of known size (for example 10 nm) through the focus, one frame per
   step.
2. Open the stack in ``Picasso: Localize`` and set ``Box side length`` and
   ``Min. net gradient`` as above, so that the beads are boxed over the
   whole stack.
3. Select ``Calibration > Calibrate astigmatism (Gaussian)``, enter the
   ``Calibration step size (nm)`` and confirm.
4. Save the calibration as ``<movie>_3d_calib.yaml``. A check plot is saved
   next to it as a ``.png``: in its panel ``Estimated z vs stage position``,
   the points should follow the diagonal over the range of z you want to
   image.

**Localize in 3D (each measurement)**

1. In ``Analyze > Parameters...``, click ``Load calibration`` in the 3D group
   and select the ``_3d_calib.yaml`` file. ``Fit Z`` is ticked automatically.
2. Keep the ``Magnification factor`` at its default of 0.79 unless you know
   the value for your sample. It corrects z for the different refractive
   indices of the immersion oil and the sample (`Huang et al., Science, 2008
   <https://doi.org/10.1126/science.1153529>`__).
3. Keep the model ``2D elliptical Gaussian`` and localize as usual. The
   localizations get an extra ``z`` column, in nm.

.. tip::

   - The cylindrical lens also slightly shifts and distorts x and y; see
     :doc:`/localize/lateral-correction` for correcting this.
   - For the best 3D precision, fit an experimentally measured PSF instead of
     a Gaussian; see :doc:`/localize/spline`.

More: :doc:`/localize`, :doc:`/localize/identification` and
:doc:`/localize/3d-calibration`.

.. _first-steps-filter:

4 · Filter (optional)
---------------------

Some localizations are of poor quality, for example two molecules blinking
so close together that they were fitted as one wide spot. :doc:`/filter`
removes them by their properties.

1. Start ``Picasso: Filter`` and drag the ``_locs.hdf5`` file into the window.
   The localizations are shown as a table.
2. Click a column header, for example ``sx``, and select
   ``Plot > Histogram`` (:kbd:`Ctrl+H`).
3. In the histogram, drag with the left mouse button over the range of values
   to keep. Localizations outside the range are removed right away.
4. Repeat for other columns if needed, then save with ``File > Save``
   (``<movie>_locs_filter.hdf5``).

Two filters that are often useful:

- ``sx`` and ``sy``: very wide spots are often two molecules on top of each
  other. Keep the main peak of the histogram.
- ``lpx`` and ``lpy``: remove the localizations with the worst precision,
  which come from dim spots.

.. figure:: /images/first-steps-filter.png
   :width: 520px
   :alt: Picasso Filter histogram of the sx column after filtering, a single peak around 0.87 pixels with the values kept between 0.75 and 1.05

   The ``sx`` histogram of the example data after keeping the values between
   0.75 and 1.05 pixels: the main peak stays, the widest and narrowest spots
   are removed. The title shows how many localizations are left.

More: :doc:`/filter`.

.. _first-steps-render:

5 · Render
----------

:doc:`/render` shows the super-resolution image built from the localizations,
and is where most of the analysis happens.

Open and explore the image
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Start ``Picasso: Render`` and drag the ``.hdf5`` file into the window.
2. Zoom in by dragging a rectangle with the left mouse button; pan by
   dragging with the right mouse button. ``View > Fit image to window``
   (:kbd:`Ctrl+W`) shows the whole image again.
3. If the image is too dark or too bright, adjust ``Min. density`` and
   ``Max. density`` in ``View > Display settings...``.

.. figure:: /images/first-steps-render.png
   :width: 520px
   :alt: Picasso Render window zoomed in on one structure of the example data; its binding sites appear as separate bright clusters, with a 20 nm scale bar

   The example data in Render, zoomed in on a single 20 nm grid DNA origami, whose binding
   sites appear as separate clusters. The localizations shown here are already
   drift-corrected (see below); before the correction, each structure looks
   smeared.

Correct the drift
~~~~~~~~~~~~~~~~~

A DNA-PAINT acquisition takes minutes to hours, and during that time the
sample slowly moves (drifts) up to thousands of nanometers. Without correction, each
structure is smeared along the path of the drift.

1. Select ``Postprocess > Undrift by AIM...`` and keep the default settings.
2. When the calculation is done, the estimated drift is plotted and the image
   is updated. The drift curves should be smooth; tens of nanometers over
   the movie are common.
3. Save the corrected localizations with ``File > Save localizations...``
   (``<file>_render.hdf5``). Use this file for all further analysis.

The drift is also saved as a ``.txt`` file, and ``Postprocess > Undo drift``
reverts the correction.

More: :doc:`/render/drift`, including a second correction round using
structures in the image as markers, which can refine the result further.

Pick regions of interest
~~~~~~~~~~~~~~~~~~~~~~~~

Picking selects the localizations of individual structures, for example single
DNA origami, or single binding sites, for closer inspection and further analysis.

1. Select ``Tools > Pick``. The cursor becomes a circle.
2. In ``Tools > Tools settings...``, set the ``Diameter (nm)`` of the circle
   so that it covers what you want to pick (the default is 100 nm).
3. Left-click on the structures you want to pick. A right click inside a pick
   removes it.
4. (Optional) ``Tools > Pick similar`` picks all other structures that look
   like the ones you picked (similar number of localizations and size).
5. Save the localizations inside the picks with
   ``File > Save picked localizations...`` (``<file>_picked.hdf5``). Each
   localization gets a ``group`` column with the number of its pick.

.. figure:: /images/first-steps-render-pick.png
   :width: 520px
   :alt: Picasso Render window showing the structure from above with a yellow circular pick around each of its ten binding sites

   Circular picks (yellow) on the binding sites of the structure shown above.
   The pick diameter was set smaller than the default, so that each pick
   covers a single binding site.

More: :doc:`/render/picking`.

Export an image
~~~~~~~~~~~~~~~

``File > Export current view...`` saves the image as shown, for example as a
``.png``.

.. _first-steps-next:

6 · Where to go next
--------------------

Each module has many more options than shown here:

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Localize
      :link: /localize
      :link-type: doc
      :class-card: sd-card-hover

      Other PSF models, GPU fitting, camera calibration, multichannel data.

   .. grid-item-card:: Filter
      :link: /filter
      :link-type: doc
      :class-card: sd-card-hover

      2D histograms, numerical filters, reapplying filters.

   .. grid-item-card:: Render
      :link: /render
      :link-type: doc
      :class-card: sd-card-hover

      Display settings, 3D view, qPAINT, multiple channels.

   .. grid-item-card:: Clustering and molecular mapping
      :link: /render/analysis
      :link-type: doc
      :class-card: sd-card-hover

      Clustering, RESI and G5M.

   .. grid-item-card:: Average
      :link: /average
      :link-type: doc
      :class-card: sd-card-hover

      Average picked structures into one particle.

   .. grid-item-card:: SPINNA
      :link: /spinna
      :link-type: doc
      :class-card: sd-card-hover

      Protein oligomerization.

In most windows, the :octicon:`book` buttons and ``File > Help`` open the
matching section of this documentation. The information saved with each file
(pixel size, processing history, etc) is explained in
:ref:`files-metadata-settings`.

Picasso as a package
--------------------

Everything the GUI does is available in Python as well. For example,
link localizations into binding events and compute their dark times:

.. code-block:: python

   from picasso import io, postprocess

   locs, info = io.load_locs("testdata_locs.hdf5")

   # Link localizations and calculate dark times
   linked_locs = postprocess.link(locs, info, r_max=0.05, max_dark_time=1)
   linked_locs = postprocess.compute_dark_times(linked_locs)

   print(f"Average bright time {linked_locs['n'].mean():.2f} frames")
   print(f"Average dark time {linked_locs['dark'].mean():.2f} frames")

More examples are in the :doc:`/notebooks` and the
:doc:`/api/index`. Some steps also run from the :doc:`/cmd`.
