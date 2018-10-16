CMD
===

Localize
--------

It is possible to reconstruct images via command line. Type:
``python -m picasso localize args`` within an environment or
``picasso localize args`` if Picasso is installed. ## Batch process a
folder To batch process a folder simply type the foldername (or drag in
drop into console), e.g. ``python -m picasso localize foldername``.
Picasso will analyze the folder and process all *.ome.tif in files in
the folder. If the files have consecutive names (e.g. File.ome.tif,
File_1.ome.tif, File_2.ome.tif) they will be treated as one. Pass Note:
Currently, only 2D Fitting is possible. ## Adding additional arguments
The reconstruction parameters can be specified by adding respective
arguments. If they are not specified the default values are chosen.*
‘-b’, ‘–box-side-length’, type=int, default=7, help=‘box side length’ \*
‘-a’, ‘–fit-method’, choices=[‘mle’, ‘lq’, ‘avg’], default=‘mle’ \*
‘-g’, ‘–gradient’, type=int, default=5000, help=‘minimum net gradient’
\* ‘-d’, ‘–drift’, type=int, default=1000, help=‘segmentation size for
subsequent RCC, 0 to deactivate’ \* ‘-bl’, ‘–baseline’, type=int,
default=0, help=‘camera baseline’ \* ‘-s’, ‘–sensitivity’, type=int,
default=1, help=‘camera sensitivity’ \* ‘-ga’, ‘–gain’, type=int,
default=1, help=‘camera gain’ \* ‘-qe’, ‘–qe’, type=int, default=1,
help=‘camera quantum efficiency’

Note 1: Localize will automatically try to perform an RCC drift
correction on the dataset. As this will not always work with the default
settings after an unsuccessful, the program will continue with the next
file. If the drift corrections succeeds, another \*.hdf5 file with the
drift corrected locs will be created.

Note 2: Make sure to set the camera settings correctly, otherwise Photon
counts are wrong plus the MLE might have problems.

Example
~~~~~~~

This example shows the batch process of a folder, with movie ome.tifs
that are supposed to be reconstructed and drift corrected with the
‘lq’-Algorithm an a gradient of 4000.

``python -m picasso localize foldername -a lq -g 4000`` or
``picasso localize foldername -a lq -g 4000``

csv2hdf
-------

Convert csv files (thunderSTORM) to hdf. Type
``python -m picasso csv2hdf filepath pixelsize``. Note that the
following columns need to be present: \* frame, x_nm, y_nm, sigma_nm,
intensity_photon, offset_photon, uncertainty_xy_nm for 2D files \*
frame, x_nm, y_nm, z_nm, sigma1_nm, sigma2_nm, intensity_photon,
offset_photon, uncertainty_xy_nm for 3D files