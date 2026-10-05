.. _render-menu-view:

View Menu
=========

.. _render-menu-display-settings:

Display settings
----------------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+D`

Opens the Display Settings Dialog, see :ref:`render-display-settings`.

.. _render-files:

Files
-----

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+F`

Opens the **Datasets** dialog, which lists every loaded channel and lets you
control its title, visibility, color (or colormap), and relative intensity. A
small horizontal gradient next to each channel previews what that channel will
look like at intensity 0 → intensity 1.

**Left click** on a channel's checkbox ticks/unticks it, i.e., shows or hides
that channel. **Right click** on a checkbox displays that channel only - it is
ticked and all other channels are unticked at once, which is convenient for
quickly inspecting individual channels in multiplexed data.

Each channel's *Color* dropdown is organized into three sections:

* **Solid colors** — the 14 default named colors (``red``, ``cyan``,
  ``green``, …). You can also type a hexadecimal code such as ``#FF5733``
  directly into the dropdown. Solid colors are rendered as a black → color
  ramp, exactly matching the previous "intensity × RGB" behavior.
* **Built-in colormaps** — one 3-stop *black → color → white* gradient per
  default solid color, named ``<color>_gradient`` (e.g. ``blue_gradient``,
  ``red_gradient``).
* **Custom** — any user-defined colormaps (see
  :ref:`render-custom-colormaps` below). This section only appears once at
  least one custom colormap has been defined.

Channels are blended additively in the final image and clipped to 1.0, so
overlapping high-intensity regions saturate toward the sum of the channel
colors.

The ``Automatic coloring`` checkbox overrides per-channel selections with
HSV-spaced colors for as long as it's ticked. ``Save colors`` /
``Load colors`` write / read a one-identifier-per-line ``.txt`` file — any
name from the three dropdown sections (or a hex code) is valid.

.. _render-custom-colormaps:

Edit custom colormaps
~~~~~~~~~~~~~~~~~~~~~

Opens a small editor where you can create, rename, duplicate, or delete your
own colormaps. Each custom colormap is a list of 2-5 *stops*; each stop has a
position in [0, 1] and an RGB color. Stops are linearly interpolated into the
256-row look-up table (LUT) used at render time.

- Click any of the R / G / B cells to type a value, or **double-click** the
  row to pick the stop color from a standard color dialog.
- Use ``Add stop`` / ``Remove stop`` to grow or shrink the gradient.

.. _render-colormaps-programmatic:

Programmatic use
~~~~~~~~~~~~~~~~

The underlying conversion from solid colors or stops to a ``(256, 3)`` LUT is
also exposed as part of ``picasso.render``:

.. code-block:: python

    from picasso import render
    lut_red   = render.solid_to_lut((1.0, 0.0, 0.0))     # black → red
    lut_fire  = render.stops_to_lut([(0, 0, 0, 0),
                                     (0.5, 1, 0, 0),
                                     (1, 1, 1, 0)])      # black → red → yellow
    qimage, *_ = render.render_scene(
        locs=..., info=..., colors=[lut_red, lut_fire], ...
    )

Passing a list of LUTs to ``render_scene`` selects the per-channel colormap
path; passing a list of plain RGB triplets (legacy) still works and is
equivalent to ``solid_to_lut`` per channel.

.. _render-overlay-image:

Overlay image
-------------

Overlays a PNG or TIFF image (``.png``, ``.tif``, ``.tiff``), e.g., a
widefield or brightfield image of the same field of view, on the rendered
localizations.

- Grayscale images of any data type (e.g., 8- or 16-bit integers or 32-bit
  floats) and RGB images are supported, with or without an alpha channel; RGB
  images with more than 8 bits per channel are scaled to 8 bits.
- For a multi-page TIFF, e.g., a raw movie, the first page is shown and the
  *Page* box selects another one; the contrast is kept when changing pages.
- Opening the dialog without a loaded image asks for one right away; an image
  can also be dropped onto the Render window.

.. _render-overlay-placement:

Placement
~~~~~~~~~

The dialog shows the size of the image and the size of the camera chip given
by the localizations' metadata (``Width`` and ``Height``), states whether they
match, and reports the resulting size of one image pixel in camera pixels and
nm. The image is placed on the camera chip using one of the following
scalings:

* **Fit to camera (keep aspect ratio)** (default) - the largest uniform
  scaling at which the whole image fits on the chip; the image is centered.
  For an image with the chip's size, this places each image pixel onto one
  camera pixel.
* **Stretch to camera** - width and height are scaled independently so that
  the image covers the whole chip. The image is distorted if its aspect ratio
  differs from the chip's.
* **Image pixel size** - each image pixel is scaled to the given pixel size
  (nm); the top left corners of the image and the chip coincide.

In every mode, the image can additionally be shifted by a given number of
camera pixels in x and y, e.g., to correct a known offset between the cameras.

The placement follows the localization coordinates: a localization at
``x = 0`` lies at the center of the first camera pixel, so the chip spans from
-0.5 to ``Width`` - 0.5 camera pixels. An image acquired on the same camera
region thus registers with the localizations to the sub-pixel level.

.. _render-overlay-display:

Display
~~~~~~~

Under *Display*, the overlay can be hidden, its opacity is set, and the
blending with the localizations is chosen:

*Additive* (default)
   Sums image and localizations.
*Over localizations*
   Paints the image over the localizations.
*Behind localizations*
   Paints the localizations over the image, so that the image shows where
   there are no localizations and shows through dim ones. A pixel is opaque at
   the maximum contrast and transparent without localizations; in between, its
   opacity is its color's distance from the color of empty pixels relative to
   the color at the maximum contrast, both given by the colormap, the
   background color and whether the background is white.
*Multiply*
   Multiplies them (for a white background).

A grayscale image is shown in the chosen color between the minimum and maximum
intensity (by default, the image's full range; *Reset contrast* restores it).
RGB images are shown with their own colors.

.. tip::

   The API functions are available as
   ``picasso.render.load_overlay_image``, ``overlay_extent``,
   ``overlay_to_qimage`` and ``draw_image_overlay``.

.. _render-fit-image:

Fit image to window
-------------------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+W` or :kbd:`Home`

Fits the reconstructed image to be fully displayed in the window.

.. _render-slice:

Slice (3D)
----------

Opens the slicer dialog which allows for slicing through 3D datasets.

.. _render-3d-view:

3D view
-------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+Shift+R`

Opens/updates the rotation window, see :doc:`3d`: with a single picked region
of interest it shows that pick, otherwise the current field of view. Requires
localizations with z coordinates.

.. _render-menu-show-info:

Show info
---------

Shows info for the current dataset. See :ref:`render-show-info`.

.. _render-new-linked-window:

New linked window
-----------------

Opens another, complete Render window with its own channels, for example to
compare channels side by side.

- A dialog asks which of the current window's channels the new window starts
  with. They are taken as they are, including unsaved changes such as
  filtering or drift correction.
- By default both windows share them in memory: filtering, drift correction,
  the Move tool, etc. in either window update both.
- More files can be opened in the new window later; they belong to that window
  only.
- By default the linked windows pan and zoom together; windows of different
  sizes show the same center at the same scale.
- Closing a linked window leaves the others open; closing the first window
  closes all of them.

.. _render-link-settings:

Link settings
-------------

Chooses which attributes the linked windows share:

- the localizations of shared channels (switching this off gives each window
  its own copy of them),
- pan, zoom, and a crosshair marking the cursor position of the window the
  mouse is in,
- the active tool,
- render settings, contrast, colormap, render by property, scale bar,
  minimap, background and legend, camera pixel size,
- pick shape and size, and the picks themselves,
- slicing (the slice position is matched in nm).

Channel colors and visibility are never shared.

When an attribute is switched on, the other windows take it from the window
the dialog was opened from; shared localizations, however, apply only to
channels taken into a linked window while the option is on.

The dialog also lists the linked windows and can bring one to the front or
unlink it; an unlinked window keeps copies of the channels it shared. The
choice is saved for the next session.
