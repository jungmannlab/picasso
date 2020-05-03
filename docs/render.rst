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
2. A dialog will appear asking for the segmentation parameter. Although the default value, 1,000 frames, is a sensible choice for most movies, it might be necessary to adjust the segmentation parameter of the algorithm, depending on the total number of frames in the movie and the number of localizations per frame. A smaller segment size results in better temporal drift resolution but requires a movie with more localizations per frame.
3. After the algorithm finishes, the estimated drift will be displayed in a pop-up window, and the display will show the drift-corrected image.

Marker-based drift correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. In ``Picasso: Render``, pick drift markers as described in ``Picking of regions of interest``. Use the ``Pick similar`` option to automatically detect a large number of drift markers similar to a few manually selected ones.
2. If the structures used as drift markers have an intrinsic size larger than the precision of individual localizations (e.g., DNA origami, large protein complexes), it is critical to select a large number of structures. Otherwise, the statistic for calculating the drift in each frame (the mean displacement of localization to the structure's center of mass) is not valid.
3. Select ``Postprocess > Undrift from picked`` to compute and apply the drift correction.
4. (Optional) Save the drift-corrected localizations by selecting ``File > Save localizations``.

Picking of regions of interest
------------------------------

1. Manual selection. Open ``Picasso: Render`` and load the localization HDF5 file to be processed.
2. Switch the active tool by selecting ``Tools > Pick``. The mouse cursor will now change to a circle. Alternatively, open ``Tools > Tools Settings`` to change the shape into a rectangle.
3. Set the size of the pick circle by adjusting the ``Diameter`` field in the tool settings dialog (``Tools > Tools Settings``). Alternatively, choose ``Width`` for a rectangular shape.
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
Adjust the general display settings.

Zooom
+++++
Set the magnification factor.

Oversampling
++++++++++++
Set the oversampling. Choose ``dynamic`` to automatically adjust to current window size when zooming.

Minimap
+++++++
Click ``show minimap`` to display a minimap in the upper left corner to localize where the current field of view is within the image.

Contrast
^^^^^^^^
Define the minimum and maximum density of the and select a colormap. Available colormaps are ['gray', hot', 'inferno', 'magma', 'plasma', 'viridis']. The selected colormap will be saved when closing render.

Blur
^^^^
Select a blur method. Available options are:
* None
* One-Pixel-Blur
* Individual Localization Precision
* Individual Localization Precision, iso

Camera
^^^^^^
Select the pixel size of the camera. This will be automatically set to a default value or the value specified in the *.yaml file.

Scale Bar
^^^^^^^^^
Activate scalebar. The length of the scale bar is calculated with the Pixel Size set in the Camera dialog. Activate  ``Print scale bar length`` to additionally print the length.

Render properties
^^^^^^^^^^^^^^^^^
This allows rendering properties by color.

Show Info
~~~~~~~~~
Displays the info dialog.

Display
^^^^^^^
Shows the image width/height, the coordinates, and dimensions of the current FoV.

Movie
^^^^^
Displays the median fit precision of the dataset. Clicking on ``Calculate`` allows calculating the precision via the NeNA approach. See `DOI: 10.1007/s00418-014-1192-3 <https://doi.org/10.1007/s00418-014-1192-3>`_.

Field of view
^^^^^^^^^^^^^
Shows the number of localizations in the current FoV.

Picks
^^^^^
Allows calculating statistics about the picked localizations. Press ``Calculate info below`` to calculate. ``Ignore dark times`` allows treating consecutive localizations as on, even if there are localizations (specified by the parameter) missing between them. When defining the number of units per pick, you can calibrate the influx rate via ``Calibrate influx``. A histogram of the dark and bright time can be plotted when clicking ``Histograms``. 


Menu items
----------

File
~~~~

Open [Ctrl+O]
^^^^^^^^^^^^^
Open an .hdf5 file to open in render.

Save localizations [Ctrl+S]
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Save the localizations that are currently loaded in render to an hdf5 file.

Save picked localizations [Ctrl+Shift+S]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Save the localizations that are within a picked region (yellow circle or rectangle). Each pick will get a different group number. To display the group number in Render, select ``Annotate picks`` in Tools/Tools Settings.
In case of rectangular picks, the saved localizations file will contain new columns `x_pick_rot` and `y_pick_rot`, which are localization coordinates into the coordinate system of the pick rectangle (coordinate (0,0) is where the rectangle was started to be drawn, and `y_pick_rot` is in the direction of the drawn line.)
These columns can be used to plot density profiles of localizations along the rectangle dimensions easily (e.g., with "Filter").

Save pick properties
^^^^^^^^^^^^^^^^^^^^
Calculates the properties of each pick (i.e., mean frame, mean x mean y as well as kinetic information and saves it as an hdf5 file.

Save pick regions
^^^^^^^^^^^^^^^^^
Saves the positions of the picked regions (yellow circles) in a .yaml file. The file will contain the following: A list of center positions and the value of the diameter. It is possible to manually add center positions or copy from another pick regions file with a text editor.

Load pick regions
^^^^^^^^^^^^^^^^^
Resets the current picked regions and loads regions from a .yaml file that contains pick regions.

Export localizations
^^^^^^^^^^^^^^^^^^^^
Select export for various other programs. Note that some exporters only work for 3D files (with z coordinates). For additional file converters check out the convert folder at Picasso's GitHub page.

Export as .csv for ThunderSTORM
+++++++++++++++++++++++++++++++

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
++++++++++++++++++++++
Export as .txt file to be used for the fourier ring correlation plugin in ImageJ.

Export as .txt for IMARIS
+++++++++++++++++++++++++
Export as .txt file to be used for IMARIS import.

Export as .xyz for Chimera
++++++++++++++++++++++++++
Export as .txt file to be used for Chimera import.

Export as .3d for ViSP
++++++++++++++++++++++
Export as .3d file to be used ViSP.

View
~~~~

Display settings (CTRL + D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Opens the Display Settings Dialog.

Files (CTRL + F)
^^^^^^^^^^^^^^^^
Open a dialog to select the color and toggle visibility for each loaded dataset.

Left / Right / Up / Down
^^^^^^^^^^^^^^^^^^^^^^^^
Moves the current field of view in a particular direction. Also possible by using the arrow keys.

Zoom in (CTRL +)
^^^^^^^^^^^^^^^^
Zoom into the image.

Zoom out (CTRL -)
^^^^^^^^^^^^^^^^^
Zoom out of the image.

Fit image to window
^^^^^^^^^^^^^^^^^^^
Fits the reconstructed image to be fully displayed in the window.

Slice (3D)
^^^^^^^^^^
Opens the slicer dialog which allows for slicing through 3D datasets.

Show info
^^^^^^^^^
Shows info for the current dataset. See Info Dialog.


Tools
~~~~~

Zoom (CTRL + Z)
^^^^^^^^^^^^^^^
Selects the zoom tool. The mouse can now be used for zoom and pan.

Pick (CTRL + P)
^^^^^^^^^^^^^^^
Selects the pick tool. The mouse can now be used for picking localizations. The user can set the pick shape in the `Tools settings` (CTRL + T) dialog. The default shape is Circle with the diameter to be set. For rectangles, the user draws the length, while the width is controlled via a parameter for all drawn rectangles, similar to the diameter for circular picks.

Measure (CTRL + M)
^^^^^^^^^^^^^^^^^^
Selects the measure tool. The mouse can now be used for measuring distances. Left click adds a crosshair for measuring; right-click deletes the last crosshair.

Tools settings (CTRL + T)
^^^^^^^^^^^^^^^^^^^^^^^^^
Define the settings of the tools, i.e., the radius of the pick and an option to annotate each pick. For the circular picks the range of pick similar can be set.

Pick similar (CTRL + Shift + P)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Automatically identifies picks that are similar to the current picks.

Show trace (CTRL + R)
^^^^^^^^^^^^^^^^^^^^^
Shows the time trace of the currently selected pick(s).

Select picks (trace)
^^^^^^^^^^^^^^^^^^^^
Opens a dialog to that goes through all picks, displays its trace and asks to keep or discard it.

Select picks (XY scatter)
^^^^^^^^^^^^^^^^^^^^^^^^^
Opens a dialog to that goes through all picks, displays a xy-scatterplot and asks to keep or discard it.

Plot pick (XYZ scatter) (CTRL + 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Displays a 3D scatterplot of the localizations of the currently selected pick(s).

Select picks (XYZ scatter)
^^^^^^^^^^^^^^^^^^^^^^^^^^
Opens a dialog to that goes through all picks, displays an xyz-scatterplot and asks to keep or discard it.

Select picks (XYZ scatter, 4 panels)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Opens a dialog to that goes through all picks, displays four panels with an xyz-scatterplot and a top, bottom and side projection and asks to keep or discard it.

Filter picks by locs
^^^^^^^^^^^^^^^^^^^^
Allows filtering picks by the number of localizations in each pick. When clicking, a histogram of the number of localizations of all selected picks will be calculated. A lower and upper boundary can be selected to filter the picks.

Clear picks (Ctrl + C)
^^^^^^^^^^^^^^^^^^^^^^
Clears all currently selected picks.

Subtract pick regions
^^^^^^^^^^^^^^^^^^^^^^
Allows loading another pick regions file to subtract from the currently selected picks. Can be slow for a large number of picks.

Show FRET traces
^^^^^^^^^^^^^^^^
Allows showing FRET traces for picks. This requires to have an acceptor and donor dataset loaded. Both channels should be aligned (i.e., via the ``Align channels (RCC or from picked)`` function). ``Show FRET traces`` will calculate a FRET intensity when two single-molecule events in one pick occur in the same frame and display a trace for these events. The intensity is calculated as I = I_A/(I_A+I_D). Here, I_A and I_D are the photon values of the localization minus the calculated background. Only FRET events > 0 and < 1 will be displayed. 

Calculate FRET in picks
^^^^^^^^^^^^^^^^^^^^^^^
Allows calculating FRET for several picks. This requires to have an acceptor and donor dataset loaded. Both channels should be aligned (i.e., via the Align channels function). The FRET intensity is calculated when two single-molecule events in one pick occur in the same frame. The intensity is calculated as I = I_A/(I_A+I_D). Here, I_A and I_D are the photon values of the localization minus the calculated background. Only FRET events in a range of > 0 and < 1 are kept. 

After calculation, a histogram of the FRET intensities is displayed. Additionally, all localizations with a valid FRET intensity are saved in an hdf5 file. The localizations have an additional column with the FRET intensities. This allows reloading the FRET-localizations in render. To color-code for FRET-intensity, use the render properties function and select FRET. Additionally, a txt document is saved containing a list of the FRET values as it was used to display the histogram.

Note: In order to calculate meaningful FRET data, the selected picks should contain data in the donor and acceptor channel. To ensure this, a sample workflow could be as follows:
- Align the channels via ``Align channels (RCC or from picked)``
- Pick some regions in one channel (i.e., the donor channel)
- Calculate the pick properties 
- Adjust the ``Pick similar`` parameter accordingly and pick similar
- Filter in the other channel (i.e., the acceptor channel) via ``Filter picks by locs`` to have at least a minimum number of localizations
- Use the calculate FRET in picks function

Cluster in pick (k-means)
^^^^^^^^^^^^^^^^^^^^^^^^^
Allows performing k-means clustering in picks. Users can specify the number of clusters and deselect individual clusters. Picks can be kept or removed. After looping through all picks an hdf5 file with the cluster information can be saved.

Mask image
^^^^^^^^^^

Postprocess
~~~~~~~~~~~

Undrift by RCC
^^^^^^^^^^^^^^
Performs drift correction by redundant cross-correlation.

Undrift from picked (3D)
^^^^^^^^^^^^^^^^^^^^^^^^
Performs drift correction using the picked localizations as fiducials. Also performs drift correction in z if the dataset has 3D information.

Undrift from picked (2D)
^^^^^^^^^^^^^^^^^^^^^^^^
Performs drift correction using the picked localizations as fiducials. Does not perform drift correction in z even if dataset has 3D information.

Undo drift (2D)
^^^^^^^^^^^^^^^
Undo previous drift correction (only 2D part). Can be pressed again to redo.

Show drift
^^^^^^^^^^
After drift correction, a drift file is created. If the drift file is present, the drift can be displayed with this option.

Remove group info
^^^^^^^^^^^^^^^^^
Removes the group information when loading a dataset that contains group information. This will, i.e., turn the multicolor representation into a single color representation.

Unfold / Refold groups
^^^^^^^^^^^^^^^^^^^^^^
Allows to "unfold" an average to display each structure individually in a line. Also works with picks.

Unfold groups (square)
^^^^^^^^^^^^^^^^^^^^^^
Arranges an average in a square so that each structure is displayed individually

Link localizations
^^^^^^^^^^^^^^^^^^
Links consecutive localizations

Align channels (RCC or from picked)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Aligns channels to each other when several datasets are loaded. If picks are selected, the alignment will be via the center of mass of the picks; otherwise, an RCC will be used. 

Combine locs in picks
^^^^^^^^^^^^^^^^^^^^^
Combines all localizations in each pick to one.

Apply expressions to localizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This tool allows you to apply expressions to localizations.

dbscan
^^^^^^
Cluster localizations with the dbscan clustering algorithm.

hdbscan
^^^^^^^
Cluster localizations with the hdbscan clustering algorithm.

Examples
++++++++
- ``x +=1`` will shift all localization by one to the right
- ``x +=1;y+=1`` will shift all localization by one to the right and one up.

Notes
+++++
Using two variables in one statement is not supported (e.g. ``x = y``) To filter localizations use picasso filter.

Additional commands
+++++++++++++++++++
``flip x z`` will exchange the x-axis with y-axis if z localizations are present (side projection), similar for ``flip y z``.
``spiral r n`` will plot each localization over the time of the movie in a spiral with radius r and n number of turns (e.g., to detect repetitive binding), ``uspiral`` to reverse.
