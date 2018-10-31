render
======

.. image:: ../docs/render.png
   :scale: 50 %
   :alt: UML Render


Opening Files
-------------
1. Rendering of the super-resolution image: In ``Picasso: Render``, open a movie file by dragging a localization file (ending with '.hdf5') into the window or by selecting ``File > Open``. The super-resolution image will be rendered automatically. A region of choice can be zoomed into by a rectangular selection using the left mouse button. The 'View' menu contains more options for zooming and panning.
2. (Optional) Adjust rendering options by selecting ``View > Display Settings``. The field 'Oversampling' defines the number of super-resolution pixels per camera pixel. The contrast settings ``Min. Density`` and ``Max. Density`` define at which number of localizations per super-resolution pixel the minimum and maximum color of the colormap should be applied.
3. (Optional) For multiplexed image acquisition, open HDF5 localization files from other channels subsequently. Alternatively, drag and drop all HDF5 files to be displayed simultaneously.

Drift Correction
----------------
Picasso offers two procedures to correct for drift: an RCC algorithm (option A), and use of specific structures in the image as drift markers (option B). Although option A does not require any additional sample preparation, option B depends on the presence of either fiducial markers or inherently clustered structures in the image. On the other hand, option B often supports more precise drift estimation and thus allows for higher image resolution. To achieve the highest possible resolution (ultra-resolution), we recommend consecutive applications of option A and multiple rounds of option B. The drift markers for option B can be features of the image itself (e.g., protein complexes or DNA origami) or intentionally included markers (e.g., DNA origami or gold nanoparticles). When using DNA origami as drift markers, the correction is typically applied in two rounds: first, with whole DNA origami structures as markers, and, second, using single DNA-PAINT binding sites as markers. In both cases, the precision of drift correction strongly depends on the number of selected drift markers.

Redundant cross-correlation drift correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. In ``Picasso: Render``, select ``Postprocess > Undrift by RCC``.
2. A dialog will appear asking for the segmentation parameter. Although the default value, 1,000 frames, is a sensible choice for most movies, it might be necessary to adjust the segmentation parameter of the algorithm, depending on the total number of frames in the movie and the number of localizations per frame66. A smaller segment size results in better temporal drift resolution but requires a movie with more localizations per frame.
3. After the algorithm finishes, the estimated drift will be displayed in a pop-up window and the display will show the drift-corrected image.

Marker-based drift correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. In ``Picasso: Render``, pick drift markers as described in ``Picking of regions of interest``. Use the ``Pick similar`` option to automatically detect a large number of drift markers similar to a few manually selected ones.
2. If the structures used as drift markers have an intrinsic size larger than the precision of individual localizations (e.g., DNA origami, large protein complexes), it is critical to select a large number of structures. Otherwise, the statistic for calculating the drift in each frame (the mean displacement of localization to the structure's center of mass) is not valid.
3. Select ``Postprocess > Undrift from picked`` to compute and apply the drift correction.
4. (Optional) Save the drift-corrected localizations by selecting ``File > Save localizations``.

Picking of regions of interest
------------------------------

1. Manual selection. Open ``Picasso: Render`` and load the localization HDF5 file to be processed.
2. Switch the active tool by selecting ``Tools > Pick``. The mouse cursor will now change to a circle.
3. Set the size of the pick circle by adjusting the ``Diameter`` field in the tool settings dialog (``Tools > Tools Settings``).
4. Pick regions of interest using the circular mouse cursor by clicking the left mouse button. All localizations within the circle will be selected for further processing.
5. (Optional) Automated region of interest selection. Select ``Tools > Pick similar`` to automatically detect and pick structures that have similar numbers of localizations and RMS deviation (RMSD) from their center of mass than already-picked structures. The upper and lower thresholds for these similarity measures are the respective standard deviations of already-picked regions, scaled by a tunable factor. This factor can be adjusted using the field ``Tools > Tools Settings > Pick similar ± range``. To display the mean and standard deviation of localization number and RMSD for currently picked regions, select ``View > Show info`` and click ``Calculate info below``.
6. (Optional) Exporting of pick information. All localizations in picked regions can be saved by selecting ``File > Save picked localizations``. The resulting HDF5 file will contain a new integer column ``group`` indicating to which pick each localization is assigned.
7. (Optional) Statistics about each pick region can be saved by selecting ``File > Save pick properties``. The resulting HDF5 file is not a localization file. Instead, it holds a data set called ``groups`` in which the rows show statistical values for each pick region.
8. (Optional) The picked positions and diameter itself can be saved by selecting ``File > Save pick regions``. Such saved pick information can also be loaded into ``Picasso: Render`` by selecting ``File > Load pick regions``.


Dialogs
-------

Display Settings
~~~~~~~~~~~~~~~~
Allows to change the display settings. Open via ``View > Display Settings``.

General
^^^^^^^

Contast
^^^^^^^

Blur
^^^^

Camera
^^^^^^

Scale Bar
^^^^^^^^^
Activate scalebar. The length of the scalebar is calculated with the Pixel Size set in the Camera dialog. Activate  ``Print scale bar length`` to additionally print the length.

Render properties
^^^^^^^^^^^^^^^^^
This allows to render properties by color.

Info
~~~~


Menu items
----------

File
~~~~

Open [Ctrl+O]
^^^^^^^^^^^^^
Open an .hdf5 file to open in render.

Save localizations [Ctrl+S]
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Save the localizations that are currently loaded in render to an .hdf5 file. 

Save picked localizations [Ctrl+Shift+S]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Save the localizations that are within a picked region (yellow circle). Each pick will get a different group number. To display the group number in Render, select ``Annotate picks`` in Tools/Tools ettings.

Save pick properties
^^^^^^^^^^^^^^^^^^^^
Calculates the properties of each pick (i.e. mean frame, mean x mean y as well as kinetic information and saves it as an .hdf5 file.

Save pick regions
^^^^^^^^^^^^^^^^^
Saves the positions of the picked regions (yellow circles) in a .yaml file. The file will contain the following: A list of center positions and the value of the diameter. It is possible to manually add center positions or copy from another pick regions file with a text editor.

Load pick regions
^^^^^^^^^^^^^^^^^
Resets the current picked regions and loads regions from a .yaml file that contains pick regions.

Export as .csv for ThunderSTORM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This will export the dataset in a .csv file to use with ThunderSTORM.

Note that for large datasets the writing of the file may take some time.

Note that the pixel size value that is set in Display Settings will be
used for exporting.

Thefollowing columns will be exported:
3D: id, frame, x [nm], y [nm], z [nm], sigma1 [nm], sigma2 [nm], intensity[photon], offset[photon], uncertainty_xy [nm] 
2D: id, frame, x [nm], y [nm], sigma [nm], intensity [photon], offset [photon], uncertainty_xy [nm]

The uncertainty_xy is calculated as the mean of lpx and lpy. For 2D, sigma is calculated as the mean of sx and sy.

For the case of linked localizations, a column named ``detections`` will be added, which contains the len parameter - that’s the duration of a blinking event and not the number n of linked localizations. This is meant to be better for downstream kinetic analysis. For a gradient that is well-chosen n ~ len and for a gap size of 0 len = n.

Export as .txt for FRC
^^^^^^^^^^^^^^^^^^^^^^
Export as .txt file to be used for fourier ring correlation plugin in ImageJ. 

Export as .txt for IMARIS
^^^^^^^^^^^^^^^^^^^^^^^^^
Export as .txt file to be used for IMARIS import.

Export as .xyz for Chimera
^^^^^^^^^^^^^^^^^^^^^^^^^^
Export as .txt file to be used for Chimera import.

Export as .3d for ViSP
^^^^^^^^^^^^^^^^^^^^^^
Export as .3d file to be used ViSP.

View
~~~~

Display settings (CTRL + D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Opens the Display Settings Dialog.

Left / Right / Up / Down
^^^^^^^^^^^^^^^^^^^^^^^^
Moves the current field of view in the particular direction. Also possible by using the arrow keys.

Zoom in (CTRL +)
^^^^^^^^^^^^^^^^
Zoom into the image.

Zoom out (CTRL -)
^^^^^^^^^^^^^^^^^
Zoom out of the image.

Fit image to window
^^^^^^^^^^^^^^^^^^^
Fits the image to be displayed in the window.

Show drift
^^^^^^^^^^
After drift correction, a drift file is created. If the drift file is present, the drift can be displayed with this option.

Show info
^^^^^^^^^
Shows info for the currrent dataset. See Info Dialog.

Tools
~~~~~

Zoom (CTRL + Z)
^^^^^^^^^^^^^^^
Selects the zoom tool. The mouse can now be used for zoom and pan. 

Pick (CTRL + P)
^^^^^^^^^^^^^^^
Selects the pick tool. The mouse can now be used for picking localizations. The radius of the pick can be set in the `Tools settings` (CTRL + T) dialog.

Measure (CTRL + M)
^^^^^^^^^^^^^^^^^^
Selects the measure tool. The mouse can now be used for measuring distances. Left click adds a crosshair for measuring, right click deletes the last crosshair.


Postprocess
~~~~~~~~~~~

Apply expressions to localizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This tool allows you to apply expressions to localizations.

Examples
++++++++
- ``x +=1`` will shift all localization by one to the right
- ``x +=1;y+=1`` will shift all localization by one to the right and one up.

Notes
+++++
Using two variables in one statement is not supported (e.g. ``x = y``) To filter localizations use picasso filter.

Additional commands
+++++++++++++++++++
``flip x z`` will exchange the x axis with y axis if z localizations are present (side projection), similar for ``flip y z``.
``spiral r n`` will plot each localization over the time of the movie in a spiral with radius r and n number of turns (e.g. to detect repetitive binding), ``uspiral`` to reverse.