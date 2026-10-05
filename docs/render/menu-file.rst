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

Opens localizations that were saved via the rotation window, see
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

Save the localizations that are within a picked region (yellow circle, square,
rectangle, polygon, box or brushed area). Each pick will get a different group
number. To display the group number in Render, select ``Annotate picks`` in
Tools/Tools Settings.

In case of rectangular picks, the saved localizations file will contain new
columns ``x_pick_rot`` and ``y_pick_rot``, which are localization coordinates
into the coordinate system of the pick rectangle (coordinate (0,0) is where
the rectangle was started to be drawn, and ``y_pick_rot`` is in the direction
of the drawn line.) These columns can be used to plot density profiles of
localizations along the rectangle dimensions easily (e.g., with "Filter").

The picked regions themselves (shape, size and positions, in the same format
as a pick regions ``.yaml`` file, see :ref:`render-save-pick-regions`) can
additionally be stored in the metadata of the saved file, under the key
``Picks``. This is switched off by default, since it can add a substantial
amount of data to the metadata. To switch it on, set
``Save picks in metadata`` to ``True`` in ``~/.picasso/settings.yaml`` (also
available under File > Picasso settings in any module).

.. _render-save-pick-properties:

Save pick properties
--------------------

Calculates the properties of each pick (i.e., mean frame, mean x mean y as
well as kinetic information) and saves it as an hdf5 file.

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
contains pick regions.

.. _render-export-roi-imaris:

Export ROI for Imaris
---------------------

This function allows to export the current ROI for Imaris. Note that this is
currently only implemented for Windows.

1. Click on File / Export ROI for imaris and enter a filename for export.
   Picasso will export the current region of interest with the current display
   pixel size settings. If multiple channels are loaded it will export the
   channels with the same colors as set in Picasso (:kbd:`Ctrl+F` or
   View / Files to change, see :ref:`render-files`).
2. Depending on the size of the ROI, the export will take a couple of seconds.
   Once exporting is finished, the file will be saved at the set location.
3. The resulting file can be opened e.g. with ImarisViewer or Imaris. Note that
   the orientation is the same as in Picasso.

.. _render-export-localizations:

Export localizations
--------------------

Select export for various other programs. Note that some exporters only work
for 3D files (with z coordinates). For additional file converters check out
the convert folder at Picasso's GitHub page.

.. _render-export-thunderstorm:

Export as .csv for ThunderSTORM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This will export the dataset in a .csv file to use with ThunderSTORM.

Note that for large datasets the writing of the file may take some time.

Note that the pixel size value that is set in Display Settings will be used
for exporting.

The following columns will be exported:

3D
   id, frame, x [nm], y [nm], z [nm], sigma1 [nm], sigma2 [nm],
   intensity[photon], offset[photon], uncertainty_xy [nm]
2D
   id, frame, x [nm], y [nm], sigma [nm], intensity [photon], offset
   [photon], uncertainty_xy [nm]

The uncertainty_xy is calculated as the mean of lpx and lpy. For 2D, sigma is
calculated as the mean of sx and sy.

For the case of linked localizations, a column named ``detections`` will be
added, which contains the len parameter - that's the duration of a blinking
event and not the number n of linked localizations. This is meant to be better
for downstream kinetic analysis. For a gradient that is well-chosen n ~ len
and for a gap size of 0 len = n.

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

Removes all .hdf5 files loaded, restarts the render window.
