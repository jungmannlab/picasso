.. _render-menu-file:

File Menu
=========

.. _render-open:

Open
----

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+O`

Open a localization file in render. Picasso ``.hdf5`` files are loaded
directly; ThunderSTORM ``.csv`` and SMAP ``_sml.mat`` files are imported (you
will be asked for the camera pixel size in nm). Localization files can also be
imported by dragging and dropping them onto the render window.

.. _render-open-rotated:

Open rotated localizations
--------------------------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+Shift+O`

Opens localizations that were saved via the 3D view, see
:doc:`3d`.

.. _render-save-localizations:

Save localizations
------------------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+S`

Save the localizations that are currently loaded in render to an hdf5 file.

.. _render-save-picked-localizations:
.. _render-save-picks-in-metadata:

Save picked localizations
-------------------------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+Shift+S`

Save the localizations that are within the picks (see
:ref:`render-pick-shapes`). The localizations get a new
integer column ``group`` with the index of their pick (see
:ref:`Table 1 <files-localization-columns>`), so each pick is drawn in its own
color when the file is opened (see :ref:`render-coloring`). To display the group number in Render, select ``Annotate picks`` in
``Tools > Tools settings`` (see :ref:`render-tools-settings`).

In case of rectangular picks, the saved localizations file will contain new
columns ``x_pick_rot`` and ``y_pick_rot``, which are localization coordinates
into the coordinate system of the pick rectangle (coordinate (0,0) is where
the rectangle was started to be drawn, and ``y_pick_rot`` is in the direction
of the drawn line.) These columns can be used to plot density profiles of
localizations along the rectangle dimensions easily.

The picked regions themselves (shape, size and positions, in the same format
as a pick regions ``.yaml`` file, see :ref:`render-save-pick-regions`) can
additionally be stored in the metadata of the saved file, under the key
``Picks``. This is switched off by default, since it can add a substantial
amount of data to the metadata. To switch it on, set
the key ``Save picks in metadata`` to ``True`` in the
:ref:`user settings file <user-settings-file>` (``~/.picasso/settings.yaml``,
also editable via ``File > Picasso settings`` in any module).

.. _render-save-picked-separately:

Save picked localizations separately
------------------------------------

Like *Save picked localizations*, but saves each pick to its own file
(``<name>_0.hdf5``, ``<name>_1.hdf5``, ...), e.g., to process single
structures individually. With more than 10 picks, Render asks for
confirmation first. With several channels, all channels can be saved one by
one (with a suffix you enter) or combined into one file per pick.

.. _render-save-pick-properties:

Save pick/group properties
--------------------------

Calculates the properties and the binding kinetics of each pick, including
the qPAINT number of binding sites, and saves them as a ``.hdf5`` file with
one row per pick (``postprocess.pick_properties``). The file can be inspected
and plotted in :doc:`/filter`, like a localization file.

For each pick:

1. The localizations are linked into binding events: localizations separated
   by at most ``Ignore dark times <=`` frames (set in the
   :ref:`Info dialog <render-info-picks>`) form one event.
2. The bright time of each event and the dark time before it are measured.
   Picks without at least two binding events are left out.
3. The mean bright and dark times are fitted to their cumulative
   distributions, and the number of binding sites is
   :math:`1 / (\text{influx rate} \cdot \tau_d)`, with the
   ``Influx rate`` from the Info dialog.

The columns include ``group`` (the pick number), ``locs`` (number of localizations) and
``n_events`` (number of binding events) per pick, the mean and standard
deviation of every localization column over the binding events
(``x_mean``, ``photons_std``, ...), the fitted ``length_cdf`` and
``dark_cdf``, ``qpaint_idx_cdf`` and ``n_units``; see
:ref:`Table 3 <files-pick-property-columns>` for all of them.

Without picks, the localizations are grouped by their ``group`` column
instead (e.g., after clustering).

This is the per-pick output of qPAINT: calibrate the influx rate in the
:ref:`Info dialog <render-info-picks>` first (it also shows the mean and
standard deviation of the bright and dark times and of the number of units
over the picks), then save the pick properties to get ``n_units`` for every
pick, e.g., to plot its distribution in :doc:`/filter`.

.. _render-save-pick-regions:

Save pick regions
-----------------

Saves the positions of the picked regions (yellow circles) in a .yaml file. It
is possible to manually add regions or copy them from another pick regions
file with a text editor. The file always carries a ``Shape`` key; the
remaining keys depend on it. All coordinates are in camera pixels and all
sizes in nm:

- ``Circle``: ``Centers`` (a list of ``[x, y]``) and ``Diameter (nm)``.
- ``Square``: ``Centers`` and ``Side Length (nm)``.
- ``Rectangle``: ``Center-Axis-Points`` (a list of
  ``[[x_start, y_start], [x_end, y_end]]``) and ``Width (nm)``.
- ``Polygon``: ``Vertices`` (a list of vertex lists, each closed by repeating
  its first vertex).
- ``Box``: ``Corners`` (a list of ``[[x0, y0], [x1, y1]]``, two opposite
  corners). No size is stored, since each box has its own.
- ``Brush``: ``Strokes``, a flat list in painting order, each with its own
  ``Width (nm)`` and the ``Path`` the cursor swept. Which strokes form one
  pick is not stored: it follows from their geometry and is worked out again
  when the file is loaded, so overlapping strokes always come back as a single
  pick.

For example:

.. code-block:: yaml

    Shape: Box
    Corners:
    - [[12.5, 30.1], [40.2, 55.7]]
    - [[80.0, 10.0], [95.5, 22.3]]

.. code-block:: yaml

    Shape: Brush
    Strokes:
    - Width (nm): 300.0
      Path:
      - [10.1, 20.4]
      - [10.9, 21.2]
    - Width (nm): 80.0
      Path:
      - [15.0, 25.0]

.. _render-load-pick-regions:

Load pick regions
-----------------

Resets the current picked regions and loads regions from a .yaml file that
contains pick regions. The ``.yaml`` file can also be dropped in the Render window.

.. _render-export-roi-imaris:

Export ROI for Imaris
---------------------

This function allows to export the current ROI for Imaris. Note that this is
only implemented for Windows.

1. Click on File / Export ROI for imaris and enter a filename for export.
   Picasso will export the current region of interest with the current display
   pixel size settings. If multiple channels are loaded it will export the
   channels with the same colors as set in Picasso (:kbd:`Ctrl+F` or
   View / Files to change, see :ref:`render-files`).
2. Depending on the size of the ROI, the export will take a couple of seconds.
   Once exporting is finished, the file will be saved at the set location.
3. The resulting file can be opened e.g. with ImarisViewer or Imaris. Note that
   the orientation is the same as in Picasso.

.. _render-export-images:

Export images
-------------

The rendered image can be saved as ``.png``, ``.tif``, ``.pdf`` or ``.svg``
(``.pdf`` asks for the resolution in DPI). Next to every image, a ``.yaml``
file records the field of view, the zoom, the display pixel size, the blur
method, the contrast, the colormap and the scale bar length. When rendering by property, the color bar is saved
as well (see :ref:`render-colorbar-format`).

``Export current view...`` (:kbd:`Ctrl+E`)
   Saves the image as shown, including picks, measurements, scale bar and
   overlays. Without a scale bar shown, a second image with a scale bar
   (``_scalebar``) is saved as well.
``Export complete image...`` (:kbd:`Ctrl+Shift+E`)
   Renders and saves the whole field of view at the current display pixel
   size.
``Export view manually...``
   Renders a region typed in (top-left corner, width and height in camera
   pixels) with a chosen display pixel size, minimum blur and blur method,
   independent of the window. The contrast is scaled to the new pixel size.
``Export channels in grayscale...``
   Saves the current view of every channel as a separate grayscale image,
   named after the channel's file plus a suffix you enter (default
   ``_grayscale.png``).

.. _render-export-localizations:

Export localizations
--------------------

Select export for various other programs. Note that some exporters only work
for 3D files (with z coordinates).

.. _render-export-thunderstorm:

Export as .csv for ThunderSTORM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exports the localizations as a ``.csv`` file for ThunderSTORM, with the
following columns, in this order (lengths converted from camera pixels to nm):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ThunderSTORM column
     - Picasso source
   * - ``id``
     - Running index of the localization.
   * - ``frame``
     - ``frame``
   * - ``x [nm]``, ``y [nm]``
     - ``x``, ``y``
   * - ``z [nm]`` (3D only)
     - ``z``
   * - ``sigma1 [nm]``, ``sigma2 [nm]`` (3D); ``sigma [nm]`` (2D)
     - ``sx``, ``sy``; in 2D, ``sx`` only
   * - ``intensity [photon]``
     - ``photons``, rounded down to an integer
   * - ``offset [photon]``
     - ``bg``, rounded down to an integer
   * - ``bkgstd [photon]``
     - Always 0 (not calculated by Picasso).
   * - ``uncertainty_xy [nm]``
     - Mean of ``lpx`` and ``lpy``.
   * - ``detections`` (linked localizations only)
     - ``len``, the duration of the binding event in frames, rather than the
       number ``n`` of linked localizations, which suits kinetic analysis
       better. Without gaps in the event, the two are equal.

.. _render-export-frc:

Export as .txt for FRC
~~~~~~~~~~~~~~~~~~~~~~

Export as .txt file to be used for the fourier ring correlation plugin in
ImageJ.

.. _render-export-chimera:

Export as .xyz for Chimera
~~~~~~~~~~~~~~~~~~~~~~~~~~

Export as .txt file to be used for Chimera import.

.. _render-export-visp:

Export as .3d for ViSP
~~~~~~~~~~~~~~~~~~~~~~

Export as .3d file to be used ViSP.

.. _render-export-smap:

Export as .mat for SMAP
~~~~~~~~~~~~~~~~~~~~~~~

Export the dataset as a SMAP (`https://github.com/jries/SMAP
<https://github.com/jries/SMAP>`_) ``_sml.mat`` file that can be loaded in SMAP
via File > Load. The output is named ``<file>_sml.mat`` (the ``_sml`` suffix is
required for SMAP to recognize the file).

Coordinates and localization precision are converted from camera pixels to nm
using the pixel size set in Display Settings; z and its precision (``lpz``)
are written in nm; frames are made 1-based (SMAP convention). ``lpx`` and
``lpy`` are combined into SMAP's single ``locprecnm`` field as their mean.

.. _render-remove-all-localizations:

Remove all localizations
------------------------

Removes all localizations loaded, restarts the Render window.

.. _render-file-other:

Sound notifications, Picasso settings and Help
----------------------------------------------

- ``Sound notifications`` selects the sound played when a long task finishes,
  see :ref:`sound-notifications`.
- ``Picasso settings...`` opens the user settings file in an editor, see
  :ref:`user-settings-file`.
- ``Help`` opens this documentation.
