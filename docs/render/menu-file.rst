.. _render-menu-file:

File Menu
=========

.. _render-open:

Open
----

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+O`

Open a localization file in render. Picasso ``.hdf5`` files are loaded
directly (a file written before Picasso 1.0 holds camera pixels and is converted to nm when it is opened, see :ref:`files-legacy`); ThunderSTORM ``.csv``, SMAP ``_sml.mat`` and PyME ``.h5r`` and ``.hdf`` files are imported. In both cases you may be asked for the camera pixel size in nm if the file does not record it. Localization files can also be imported by dragging and dropping them onto the render window. A PyME file with several channels (``probe`` column) opens as one channel per probe, named ``<file>_probe<k>``. See :ref:`render-export-pyme` for how the columns are converted.

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

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      from picasso import io

      io.save_locs("movie_locs_render.hdf5", locs, info)


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

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``picked_locs`` adds the ``group`` column (``add_group=True``).

   .. code-block:: python

      import pandas as pd
      from picasso import postprocess

      picked = postprocess.picked_locs(
          locs, info, picks=picks, pick_shape="Circle", pick_size=diameter
      )
      io.save_locs("movie_locs_picked.hdf5", pd.concat(picked, ignore_index=True), info)


.. _render-save-picked-separately:

Save picked localizations separately
------------------------------------

Like *Save picked localizations*, but saves each pick to its own file
(``<name>_0.hdf5``, ``<name>_1.hdf5``, ...), e.g., to process single
structures individually. With more than 10 picks, Render asks for
confirmation first. With several channels, all channels can be saved one by
one (with a suffix you enter) or combined into one file per pick.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      for i, locs_in_pick in enumerate(picked):
          io.save_locs(f"movie_locs_pick{i}.hdf5", locs_in_pick, info)


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

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``pick_areas`` (in µm²) are needed for the density columns; the circle
   size is the diameter in nm.

   .. code-block:: python

      from picasso import io, lib, postprocess

      areas = lib.pick_areas(picks, "Circle", diameter) * 1e-6  # nm² -> µm²
      props = postprocess.pick_properties(
          picked, info, max_dark_time=3, influx_rate=0.03, pick_areas=areas
      )
      io.save_datasets("movie_locs_pickprops.hdf5", info, groups=props)
      props = pd.read_hdf("movie_locs_pickprops.hdf5", key="groups")


.. _render-save-pick-regions:

Save pick regions
-----------------

Saves the positions of the picked regions (yellow circles) in a .yaml file. It
is possible to manually add regions or copy them from another pick regions
file with a text editor. The file always carries a ``Shape`` key and the
format marker ``picasso_format`` (see :ref:`files-format-versions`); the
remaining keys depend on the shape. All coordinates and all sizes are in nm:

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
    picasso_format: '1.0'
    Corners:
    - [[1625.0, 3913.0], [5226.0, 7241.0]]
    - [[10400.0, 1300.0], [12415.0, 2899.0]]

.. code-block:: yaml

    Shape: Brush
    picasso_format: '1.0'
    Strokes:
    - Width (nm): 300.0
      Path:
      - [1313.0, 2652.0]
      - [1417.0, 2756.0]
    - Width (nm): 80.0
      Path:
      - [1950.0, 3250.0]

Pick regions saved before Picasso 1.0 (no ``picasso_format`` key) hold their
coordinates in camera pixels; they are converted when loaded, see
:ref:`files-legacy`.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``io.save_picks`` writes the file (positions and size in nm);
   ``io.picks_dict`` returns the same content as a dictionary.

   .. code-block:: python

      from picasso import io

      picks = [(15600.0, 11453.0), (17030.0, 12051.0)]  # centers, nm
      io.save_picks("movie_locs_picks.yaml", picks, "Circle", size=100.0)


.. _render-load-pick-regions:

Load pick regions
-----------------

Resets the current picked regions and loads regions from a .yaml file that
contains pick regions. The ``.yaml`` file can also be dropped in the Render window.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   Positions and the size (a diameter for circles) come back in nm.
   ``pixelsize`` is needed only for files written before Picasso 1.0.

   .. code-block:: python

      picks, shape, size = io.load_picks("movie_locs_picks.yaml")


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
   the orientation is the same as in Picasso. If the localizations come from an
   ``.ims`` movie, the export is placed at the movie's position (including its
   focal plane in z), so it overlays the movie in Imaris.

.. _render-export-images:

Export images
-------------

The rendered image can be saved as ``.png``, ``.tif``, ``.pdf`` or ``.svg``
(``.pdf`` asks for the resolution in DPI). Next to every image, a ``.yaml``
file records the field of view (``FOV (X, Y, Width, Height; nm)``), the
zoom (``Zoom (nm per screen pixel)``, the nm covered by one screen pixel),
the display pixel size, the render method, the contrast, the colormap and the
scale bar length. When rendering by property, the color bar is saved as well
(see :ref:`render-colorbar-format`). Dropping such a ``.yaml`` file onto
Render restores the view; files written before Picasso 1.0 (keys ``FOV (X, Y,
Width, Height)`` in camera pixels and ``Zoom`` in screen pixels per camera
pixel) are read as well.

``Export current view...`` (:kbd:`Ctrl+E`)
   Saves the image as shown, including picks, measurements, scale bar and
   overlays. Without a scale bar shown, a second image with a scale bar
   (``_scalebar``) is saved as well.
``Export complete image...`` (:kbd:`Ctrl+Shift+E`)
   Renders and saves the whole field of view at the current display pixel
   size.
``Export view manually...``
   Renders a region typed in (top-left corner, width and height in nm)
   with a chosen display pixel size, minimum blur and render method,
   independent of the window. The contrast is scaled to the new pixel size.
``Export channels in grayscale...``
   Saves the current view of every channel as a separate grayscale image,
   named after the channel's file plus a suffix you enter (default
   ``_grayscale.png``).

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   See :ref:`render-opening-files` for the rendering options.

   .. code-block:: python

      import matplotlib.pyplot as plt
      from picasso import render

      n_locs, image = render.render(
          locs, info, disp_px_size=10, viewport=viewport, blur_method="gaussian"
      )
      plt.imsave("movie.png", image, vmin=0, vmax=0.2 * image.max(), cmap="magma")


.. _render-export-localizations:

Export localizations
--------------------

Select export for various other programs. Note that some exporters only work
for 3D files (with z coordinates).

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      locs.to_csv("movie_locs.csv", index=False)


.. _render-export-thunderstorm:

Export as .csv for ThunderSTORM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exports the localizations as a ``.csv`` file for ThunderSTORM, with the
following columns, in this order (lengths in nm, as in Picasso):

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

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``io.import_ts`` reads a ThunderSTORM file (and saves it as
   ``_locs.hdf5``). The coordinates stay in nm; ``pixelsize`` is the camera
   pixel size recorded in the metadata (``Pixelsize``), from which the field
   of view (``Width``, ``Height``) is derived.

   .. code-block:: python

      io.export_thunderstorm("movie_locs.csv", locs, info)
      locs, info = io.import_ts("thunderstorm.csv", pixelsize=130)


.. _render-export-imagej:

Export as .txt for ImageJ
~~~~~~~~~~~~~~~~~~~~~~~~~

Exports the localizations as a plain ``.txt`` file without a header, with the
columns ``frame``, ``x`` and ``y`` (in nm), for ImageJ.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      io.export_txt_imagej("movie_locs.txt", locs, info)


.. _render-export-chimera:

Export as .xyz for Chimera
~~~~~~~~~~~~~~~~~~~~~~~~~~

Export as .txt file to be used for Chimera import.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      io.export_xyz_chimera("movie_locs.chi.xyz", locs, info)


.. _render-export-visp:

Export as .3d for ViSP
~~~~~~~~~~~~~~~~~~~~~~

Export as .3d file to be used ViSP.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      io.export_3d_visp("movie_locs.visp.3d", locs, info)


.. _render-export-smap:

Export as .mat for SMAP
~~~~~~~~~~~~~~~~~~~~~~~

Export the dataset as a SMAP (`https://github.com/jries/SMAP
<https://github.com/jries/SMAP>`_) ``_sml.mat`` file that can be loaded in SMAP
via File > Load. The output is named ``<file>_sml.mat`` (the ``_sml`` suffix is
required for SMAP to recognize the file).

Coordinates, localization precisions, z and its precision (``lpz``) are
written in nm, as they are in Picasso; frames are made 1-based (SMAP
convention). ``lpx`` and ``lpy`` are combined into SMAP's single
``locprecnm`` field as their mean.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   On import, the coordinates stay in nm; ``pixelsize`` is the camera pixel
   size recorded in the metadata.

   .. code-block:: python

      io.export_smap("movie_locs_sml.mat", locs, info)
      locs, info = io.import_smap("smap_sml.mat", pixelsize=130)


.. _render-export-pyme:

Export as .hdf for PyME
~~~~~~~~~~~~~~~~~~~~~~~

Export the dataset as a PyME (`python-microscopy
<https://python-microscopy.org>`_) ``.hdf`` file that can be opened in
PYMEVisualize. It has the format of PYMEVisualize's "Save filtered
localizations", i.e., a table ``Localizations`` with lengths in nm. When
exporting all channels, they are saved in one file and told apart by PyME's
``probe`` column (PYMEVisualize shows them as ``chan0``, ``chan1``, etc.). The
channels must have the same camera pixel size.

The columns are converted as follows:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Picasso
     - PyME
     - Note
   * - ``x``, ``y``
     - ``x``, ``y``
     - nm in both
   * - ``frame``
     - ``t``
     -
   * - ``z``, ``lpz``
     - ``z``, ``error_z``
     - nm in both
   * - ``lpx``, ``lpy``
     - ``error_x``, ``error_y``
     - nm in both
   * - ``sx``, ``sy``
     - ``sigx``, ``sigy``
     - nm in both
   * - ``photons``
     - ``nPhotons``
     -
   * - ``bg``
     - ``bg``
     - photons per camera pixel
   * - channel index
     - ``probe``
     -

Other numeric columns (e.g., ``group``) are copied as they are. With several
channels, only the columns present in all of them are kept. The pixel size is
stored as ``voxelsize`` (in µm, as in PyME), and the Picasso metadata of each
channel is stored, too, so that importing the file back into Picasso restores
it.

.. note::

   PYMEVisualize filters the localizations by ``error_x`` and ``error_y``
   (below 30 nm) by default, i.e., localizations with a worse precision are
   hidden until the filter is changed. The PyME columns ``A`` and ``sig`` are
   not written, since PYMEVisualize's default filters on them (e.g., ``sig``
   between 95 and 200 nm) would hide localizations, too.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      io.export_pyme("movie_locs.hdf", locs, info)
      # several channels in one file
      io.export_pyme_multi("channels.hdf", [locs1, locs2], [info1, info2])

      # pixelsize (camera pixel size, for the metadata) is used only if the
      # file does not store it; the coordinates stay in nm
      locs, info = io.import_pyme("pyme.hdf", pixelsize=130)
      # one (path, locs, info) per probe value
      channels = io.split_pyme_channels("pyme.hdf", locs, info)


.. _render-convert-files:

Convert files to nm
-------------------

``File > Convert files to nm...`` converts files written before Picasso 1.0,
whose lengths are in camera pixels, to the current format in nm:
localizations (with their ``.yaml``), pick regions, drift ``.txt`` files,
astigmatism and lateral calibrations, SPINNA masks and nanoTRON models. Add
the files to the list, enter the camera pixel size (prefilled from the first
file's metadata if it records it, else from the last loaded channel) and
choose whether to replace the original files or to write the converted ones
next to them with ``_nm`` appended to their names. Opening an old file
converts it in memory anyway; the dialog saves the converted files on disk. See :ref:`files-legacy` and, for the command
line, :ref:`cmd-convert-units`.

.. _render-remove-all-localizations:

Remove all localizations
------------------------

Removes all localizations loaded, restarts the Render window.

To be asked for confirmation first, set ``Confirm removing all
localizations`` to ``true`` in the :ref:`user settings <user-settings-file>`;
ticking *Don't ask again* in the question sets it back to ``false``.

.. _render-file-other:

Sound notifications, Picasso settings and Help
----------------------------------------------

- ``Sound notifications`` selects the sound played when a long task finishes,
  see :ref:`sound-notifications`.
- ``Picasso settings...`` opens the user settings file in an editor, see
  :ref:`user-settings-file`.
- ``Help`` opens this documentation.
