render
======

File
----

Open [Ctrl+O]
~~~~~~~~~~~~~

Open an *.hdf5 file to open in render.

Save localizations [Ctrl+S]
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Save the localizations that are currently loaded in render to an*.hdf5 file. 

Save picked localizations [Ctrl+Shift+S]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Save the localizations that are within a picked region (yellow circle). Each pick will get a different group number. To display the group number in Render, select ‘Annotate picks’ in Tools/Tools ettings.

Save pick properties
~~~~~~~~~~~~~~~~~~~~
Calculates the properties of each pick (i.e. mean frame, mean x mean y as well as kinetic information and saves it as an *.hdf5 file.

Save pick regions
~~~~~~~~~~~~~~~~~
Saves the positions of the picked regions (yellow circles) in a*.yaml file. The file will contain the following: A list of center positions and the value of the diameter. It is possible to manually add center positions or copy from another pick regions file with a text editor.

Load pick regions
~~~~~~~~~~~~~~~~~
Resets the current picked regions and loads regions from a \*.yaml file that contains pick regions.

Export as .csv for ThunderSTORM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This will export the dataset in a .csv file to use with ThunderSTORM.

Note that for large datasets the writing of the file may take some time.

Note that the pixel size value that is set in Display Settings will be
used for exporting.

Thefollowing columns will be exported:
3D: id, frame, x [nm], y [nm], z [nm], sigma1 [nm], sigma2 [nm], intensity[photon], offset[photon], uncertainty_xy [nm] 
2D: id, frame, x [nm], y [nm], sigma [nm], intensity [photon], offset [photon], uncertainty_xy [nm]

The uncertainty_xy is calculated as the mean of lpx and lpy. For 2D, sigma is calculated as the mean of sx and sy.

For the case of linked localizations, a column named ``detections`` will be added, which contains the len parameter - that’s the duration of a blinking event and not the number n of linked localizations. This is meant to be better for downstream kinetic analysis. For a gradient that is well-chosen n ~ len and for a gap size of 0 len = n.

View
----

Tools
-----

Postprocess
-----------

Apply expressions to localizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tool allows you to apply expressions to localizations.

Examples:
^^^^^^^^^
``x +=1`` will shift all localization by one to the right
``x +=1;y+=1`` will shift all localization by one to the right and one up.

Notes:
^^^^^^
Using two variables in one statement is not supported (e.g. ``x = y``) To filter localizations use picasso filter.

Additional commands:
^^^^^^^^^^^^^^^^^^^^
``flip x z`` will exchange the x axis with y axis if z localizations are present (side projection), similar for ``flip y z``.
``spiral r n`` will plot each localization over the time of the movie in a spiral with radius r and n number of turns (e.g. to detect repetitive binding), ``uspiral`` to reverse.