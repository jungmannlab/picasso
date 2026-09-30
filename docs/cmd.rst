CMD
===

.. image:: ../docs/cmd.png
   :scale: 50 %
   :alt: UML Picasso cmd

Here is a list of command-line commands that can be used with picasso. Each command can be run by typing ``picasso command args`` in a terminal or command prompt, where ``command`` is one of the commands listed below and ``args`` are the respective arguments for that command. For more information, type ``picasso -h`` for the list of commands, or ``picasso command -h`` for the arguments of one command. That help text is generated from the code, so it is the authoritative and always up-to-date list of what each command accepts; the sections below describe behavior that does not fit into a one-line help string.

If you wish to open a module (GUI), simply type ``picasso module_name``, for example, ``picasso render``.

localize
--------
Localize identifies and fits single-molecule spots in a movie. Type ``picasso localize`` to open the GUI module, or ``picasso localize path args`` to run the analysis from the command line, where ``path`` is a movie file, a folder or a file pattern.

Finding out which arguments exist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``picasso localize -h`` prints every argument with its short and long name, its type, its default value and a one-line description. That text is generated from the code itself, so it is always complete and up to date — use it as the reference for what can be set. Arguments that are not given keep their defaults, so the shortest possible run is just ``picasso localize foldername``.

The arguments fall into a few groups:

* **Spot identification** — box side length, the identification method (``--identification-method``) with its threshold (the minimum net gradient, or the ``--wavelet-*`` settings), and the pre-filters ``--temporal-median`` and ``--gaussian-filter``.
* **Fitting** — the fit method and the calibration files that some methods require (a spline PSF calibration for the spline fits, a magnification factor and a 3D calibration for the astigmatism fits).
* **Camera** — baseline, sensitivity, gain and pixel size.
* **What is analyzed** — region of interest, frame bounds, and the flags that change how several movies or several regions are grouped (``--concat``, ``--regions-separately``).
* **Output** — drift correction segmentation, a suffix for the output files, and whether the run is added to the local database.

The rest of this section explains only those options whose behavior needs more than the one line that ``-h`` gives.

Batch process a folder
~~~~~~~~~~~~~~~~~~~~~~
To batch process a folder simply type the folder name (or drag and drop it into the console), e.g. ``picasso localize foldername``. Picasso will analyze the folder and process all *.ome.tif files in it. If the files have consecutive names (e.g., File.ome.tif, File_1.ome.tif, File_2.ome.tif), they will be treated as one.
If you want to analyze *.raw files, Picasso will check whether a *.raw file has a corresponding *.yaml file. If none is found, you can enter the specifications for each raw file. It is possible to use the same specifications for all *.raw files in that run.

Drift correction
~~~~~~~~~~~~~~~~
Localize will automatically try to perform an RCC drift correction on the dataset. As this will not always work with the default settings, after an unsuccessful attempt the program will continue with the next file. If the drift correction succeeds, another hdf5 file with the drift corrected locs will be created.

Camera settings
~~~~~~~~~~~~~~~
Make sure to set the camera settings correctly; otherwise photon counts are wrong plus the MLE might have problems.

Pre-filters
~~~~~~~~~~~
``--temporal-median`` subtracts a rolling per-pixel median background before spots are identified, which suppresses uneven background and static structures. It affects identification only. See Martens KJA, Turkowyd B, Endesfelder U, `Raw data to results: a hands-on introduction and overview of computational analysis for single-molecule localization microscopy <https://doi.org/10.3389/fbinf.2021.817254>`_, *Frontiers in Bioinformatics* 1, 817254 (2022).

``--gaussian-filter`` smooths each frame with a Gaussian of the given standard deviation before spots are identified. Spot identification looks for a single local maximum per spot, so a PSF that is not Gaussian-shaped may break into several maxima and is detected several times; smoothing merges them into one. It affects identification only — fitting always uses the raw movie — and since smoothing lowers gradient magnitudes, the minimum net gradient needs re-tuning when it is changed. It can be combined with ``--temporal-median``, which is applied first.

B-spline wavelet identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``--identification-method wavelet`` (``-im wavelet``) identifies spots by the B-spline wavelet segmentation of Izeddin et al., *Optics Express* 20, 2081 (2012), instead of by their net gradient; ``--gradient`` is then ignored. ``--wavelet-threshold`` sets the threshold in units of the noise standard deviation (default 0.5), ``--wavelet-noise`` how the noise is estimated (``image-std``, the default, or ``w1-mad``, which is robust to dense spots and uneven background) and ``--wavelet-min-area`` the smallest region kept (default 4 pixels). The localizations have no ``net_gradient`` column. See *B-spline wavelet identification* in the Localize documentation. ``spline-calibrate`` and ``lateral-calibrate`` accept the same arguments for detecting beads::

   picasso localize movie.tif -b 7 -im wavelet --wavelet-threshold 1

Analyzing several movies as one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``--concat`` accumulates the movies found into one concatenated movie. Given a folder, Picasso searches it and all of its sub-folders; given a pattern (e.g. ``"experiment_folder/*/*.tif"``), it uses the matching files. In both cases the files are ordered by folder and file name with numbers compared numerically, so ``run_2`` comes before ``run_10``, and the full order is printed before the analysis starts — check it, since a wrong order is only noticeable afterwards. Each entry is one whole movie: the continuation files of a split OME-TIFF stack (``*_1.ome.tif``, ...) and the individual frames of a MicroManager "separate image files" folder are read together with the movie they belong to, so no frames are repeated.

Analyzing regions of one movie separately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``--regions-separately`` treats the ``--roi`` regions as channels imaged side by side on one sensor and fits each of them on its own, writing ``<movie>_ref_locs.hdf5``, ``<movie>_ch1_locs.hdf5``, ... — one file per region, each in that region's own coordinates (the region's corner subtracted from ``x`` and ``y``, and its size in the metadata), so the files overlay each other when loaded as channels in Render. The fit method, the minimum net gradient and the spline calibration may then be given once per ``--roi``, in the same order as the regions, or once for all of them::

   picasso localize movie.tif -b 7 --regions-separately --roi 0 0 256 256 --roi 0 256 256 512 -g 5000 -g 2000 -a lq -a mle

Separate movies need no flag: ``picasso localize`` already analyzes each file on its own. See *Analyzing each channel on its own* in the Localize documentation.

3D fitting
~~~~~~~~~~
If you select one of the astigmatism-based 3D algorithms (``lq-3d``, ``lq-gpu-3d`` or ``mle-3d``) you must supply both the magnification factor (``-mf``) and the path to the 3D calibration file (``-zc``). If either is omitted, the program will prompt you for it interactively. The spline fit methods instead need a cubic-spline PSF calibration passed with ``--spline-calibration``.

Example
~~~~~~~
This example shows the batch process of a folder, with movie ome.tifs that are supposed to be reconstructed and drift corrected with the ``lq`` algorithm and a min. net gradient of 4000.

``picasso localize foldername -a lq -g 4000``

render
------
Start the render module (GUI) or render from command line. With no arguments the GUI opens; with a ``files`` path one or more localization files are rendered to image files.

::

   '-px', '--disp-px-size', type=float, default=1.0, help='the size of the rendered pixel in nm'
   '-b', '--blur-method', choices=['none', 'convolve', 'gaussian'], default='convolve'
   '-w', '--min-blur-width', type=float, default=0.0, help='minimum blur width if blur is applied'
   '--vmin', type=float, default=0.0, help='minimum colormap level in range 0-100 or absolute value'
   '--vmax', type=float, default=20.0, help='maximum colormap level in range 0-100 or absolute value'
   '--scaling', choices=['yes', 'no'], default='yes', help='if scaling, the colormap value is relative in the range 0-100'
   '-c', '--cmap', choices=['viridis', 'inferno', 'plasma', 'magma', 'hot', 'gray'], help='the colormap to be applied'
   '-s', '--silent', action='store_true', help='do not open the rendered image file'

filter
------
Start the filter module (GUI).

design
------
Start the design module (GUI).

simulate
--------
Start the simulation module (GUI).

average
-------
Start the 2D averaging module (GUI).

server
------
Start the Picasso server (web browser GUI).

spinna
------
Start the SPINNA module (GUI) or run batch analysis from command line. Without ``-p`` the GUI opens; passing ``-p path/to/parameters.csv`` runs batch analysis from a parameters CSV file (see the SPINNA documentation for the expected CSV structure).

::

   '-p', '--parameters', type=str, help='.csv file containing the parameters for spinna batch analysis'
   '-a', '--asynch', action='store_false', help='do not perform fitting asynchronously (multiprocessing)'
   '-b', '--bootstrap', action='store_true', help='perform bootstrapping'
   '-v', '--verbose', action='store_true', help='display progress bar for each row'

average3
--------
Start the 3D averaging module (GUI) (to be deprecated in 1.0).

csv2hdf
-------
Convert csv files (ThunderSTORM) to ``.hdf5``. Type ``picasso csv2hdf filepath -p pixelsize`` (``-p/--pixelsize`` in nm is required). Note that the following columns need to be present:
``frame, x_nm, y_nm, sigma_nm, intensity_photon, offset_photon, uncertainty_xy_nm`` for 2D files
``frame, x_nm, y_nm, z_nm, sigma1_nm, sigma2_nm, intensity_photon, offset_photon, uncertainty_xy_nm`` for 3D files

hdf2csv
-------
Convert hdf5 files to ``.csv`` files (keeps columns names).

hdf2ts
------
Convert hdf5 files to ThunderSTORM ``.csv`` files (adapts column names).

hdf2imagej
----------
Convert hdf5 files to ImageJ ``.txt`` files.

hdf2nis
-------
Convert hdf5 files to NIS Elements ``.txt`` files.

hdf2chimera
------------
Convert hdf5 files to Chimera ``.xyz`` files.

hdf2visp
--------
Convert hdf5 files to ViSP ``.3d`` files.

smap2hdf
--------
Convert SMAP (`https://github.com/jries/SMAP <https://github.com/jries/SMAP>`_) ``_sml.mat`` files to ``.hdf5``. Type ``picasso smap2hdf filepath -p pixelsize`` (``-p/--pixelsize`` in nm is required, since SMAP stores coordinates in nm while Picasso uses camera pixels). Reads single-file MATLAB ``-v7`` and ``-v7.3`` saves; split ``_sml_p*.mat`` parts files are not supported (re-save them in SMAP as a single file).

hdf2smap
--------
Convert hdf5 files to SMAP ``_sml.mat`` files. The output is named ``<file>_sml.mat`` (the ``_sml`` suffix is required for SMAP to recognize the file) and can be loaded directly in SMAP via File > Load.

join
----
Combine two hdf5 localization files. Type ``picasso join file1 file2``. A new joined file will be created. Note that the frame information of consecutive files is reindexed, i.e., frame 1 now can contain localizations from file 1 and file 2. Therefore, do not perform kinetic analysis and drift correction on joined files. Pass ``-k/--keepindex`` to keep the original frame numbers instead of reindexing. Columns that not all files have (e.g. ``z`` when joining 2D and 3D files, or ``net_gradient`` when only some were identified by their net gradient) are dropped with a warning, so that no localizations are lost.

link
----
Link localizations in consecutive frames.

::

   '-d', '--distance', type=float, default=1.0, help='maximum distance between localizations to consider them the same binding event (camera pixels)'
   '-t', '--tolerance', type=int, default=1, help='maximum dark time between localizations to still consider them the same binding event'

clusterfilter
-------------
Filter localizations by properties of their clusters.

::

   '-c', '--clusterfile', type=str, help='a hdf5 clusterfile'
   '-p', '--parameter', type=str, help='parameter to be filtered'
   '--minval', type=float, help='lower boundary'
   '--maxval', type=float, help='upper boundary'

undrift
-------
Correct localization coordinates for drift with RCC.

::

   '-s', '--segmentation', type=float, default=1000, help='the number of frames to be combined for one temporal segment'
   '-f', '--fromfile', type=str, help='apply drift from specified file instead of computing it'
   '-d', '--display', action='store_true', help='display estimated drift'

aim
---
Correct localization coordinates for drift with AIM.

::

   '-s', '--segmentation', type=float, default=100, help='the number of frames to be combined for one temporal segment'
   '-i', '--intersectdist', type=float, default=20/130, help='max. distance (camera pixels) between localizations in consecutive segments to be considered as intersecting'
   '-r', '--roiradius', type=float, default=60/130, help='max. drift (camera pixels) between two consecutive segments'

undrift_fiducials
-----------------
Correct localization coordinates for drift using fiducial markers (automatically picked). Takes one or more hdf5 localization files as positional arguments.

density
-------
Compute the local density of localizations. Takes positional ``files`` and ``radius`` (float): the maximal distance between two localizations to be considered local.

dbscan
------
Cluster localizations with the dbscan clustering algorithm. Positional arguments:

::

   files                one or more hdf5 localization files
   radius (float)       maximal distance (camera pixels) between two localizations to be considered local
   density (int)        minimum local density for localizations to be assigned to a cluster
   pixelsize (int)      camera pixel size in nm (required for 3D localizations only)

hdbscan
-------
Cluster localizations with the hdbscan clustering algorithm. Positional arguments:

::

   files                one or more hdf5 localization files
   min_cluster (int)    smallest size grouping that is considered a cluster
   min_samples (int)    the higher, the more points are considered noise
   pixelsize (int)      camera pixel size in nm (required for 3D localizations only)

smlm_cluster
------------
Cluster localizations with the custom SMLM clustering algorithm.

The algorithm finds localizations with the most neighbors within a specified radius and finds clusters based on such "local maxima".

Positional arguments:

::

   files                one or more hdf5 localization files
   radius (float)       clustering radius (in camera pixels)
   min_locs (int)       minimum number of localizations in a cluster
   pixelsize (int)      camera pixel size in nm (required for 3D localizations only)
   basic_fa (bool)      whether to perform basic frame analysis (sticking event removal)
   radius_z (float)     clustering radius in axial direction (MUST BE SET FOR 3D!)

g5m
---
Run Gaussian Mixture Modeling with Modifications (G5M) for Molecular Mapping on clustered localizations. For details see https://doi.org/10.1038/s41467-026-70198-5.

The positional ``files`` argument is a unix-style path to one or more clustered ``.hdf5`` files, or a folder in which all ``.hdf5`` files will be analyzed. If omitted, the GUI is launched.

::

   '-ml', '--min-locs', type=int, default=10, help='min. number of locs per molecule'
   '-lph', '--loc-prec-handle', type=str, default='local', help="loc. precision handle, either 'local' or 'abs'"
   '--min-sigma', type=float, default=0.8, help='minimum sigma factor/value'
   '--max-sigma', type=float, default=1.5, help='maximum sigma factor/value'
   '--max-rounds', type=int, default=3, help='max. rounds without BIC improvement to terminate'
   '--bootstrap-sem', action='store_true', help='bootstrap to estimate SEM of molecule positions'
   '-c', '--calibration', type=str, default='', help='path to calibration file (3D only)'
   '--covariance-type', type=str, choices=['auto', 'spherical', 'diagonal', 'rotated'], default='auto', help="shape of the fitted molecules; 'auto' picks 'rotated' for 3D astigmatism locs carrying an 'angle' column, 'diagonal' for 3D localizations without the 'angle' column and spherical for 2D data"
   '-p', '--postprocess', action='store_false', help='do not postprocess results to remove sticking events and low-quality fits'
   '--max-locs', type=int, default=100000, help='maximum number of localizations to process per cluster; useful for excluding fiducials'
   '-a', '--asynch', action='store_false', help='do not perform fitting asynchronously (multiprocessing)'

dark
----
Compute the dark time for grouped localizations.

align
-----
Align one localization file to antoher via RCC.
Type ``picasso align file1 file2 [...]`` (two or more files). Pass ``-d/--display`` to display the correlation.

groupprops
----------
Calculate the properties of localization groups

pc
--
Calculate the pair-correlation of localizations.

::

   '-b', '--binsize', type=float, default=0.1, help='the bin size (camera pixels)'
   '-r', '--rmax', type=float, default=10, help='the maximum distance to calculate the pair-correlation'

nneighbor
---------
Calculate the nearest neighbor within a clustered dataset.

cluster_combine
---------------
Combines the localizations in each cluster of a group (to be deprecated in 1.0).

cluster_combine_dist
--------------------
Calculate the nearest neighbor for each combined cluster (to be deprecated in 1.0).

plugins
-------
Manage Picasso plugins without opening a GUI. ``picasso plugins list`` shows every plugin file in ``~/.picasso/plugins``, whether it is enabled, and what each enabled one contributes (a GUI menu entry, command line commands, API names). ``picasso plugins enable <file.py>`` and ``disable <file.py>`` control whether a file is loaded and run at all, and ``install``, ``update`` and ``uninstall`` work against the online registry with the same hash verification as the plugin browser. ``picasso plugins path`` prints the plugins folder.

Plugins can add their own commands here: an enabled plugin that defines ``register_cli`` contributes subcommands that appear in ``picasso -h`` and behave like the built-in ones. See :doc:`plugins` for how to write one.
