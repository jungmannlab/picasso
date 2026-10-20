Command Line
============

.. figure:: /images/cmd.png
   :width: 640px
   :class: screenshot
   :alt: Command prompt window with the output of picasso -h, listing the Picasso subcommands and their descriptions

   The list of commands printed by ``picasso -h``.

Here is a list of command-line commands that can be used with Picasso. Each
command can be run by typing ``picasso command args`` in a terminal or command
prompt, where ``command`` is one of the commands listed below and ``args`` are
the respective arguments for that command.

.. code-block:: bash

   picasso -h           # the list of commands
   picasso command -h   # the arguments of one command

If you wish to open a module (GUI), simply type ``picasso module_name``, for
example:

.. code-block:: bash

   picasso render

.. _cmd-renamed:

Renamed commands and options
----------------------------
In Picasso 1.0, the commands and options got one style: words joined by
hyphens, as in ``camera-calibrate``, and named after the Python parameter
they set where there is one (``--r-max`` for ``r_max``). The old names keep
working until Picasso 2.0; each use prints a warning with the new name. The
format converters (``csv2hdf``, ``hdf2ts``, ...) keep their names. See
:doc:`versioning` for how long deprecated names keep working.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Old name
     - New name
   * - ``undrift_fiducials``
     - `undrift-fiducials`_
   * - ``nneighbor``
     - `nearest-neighbor`_
   * - ``clusterfilter``
     - `cluster-filter`_, with ``--cluster-file``, ``--min-value`` and
       ``--max-value`` (were ``--clusterfile``, ``--minval``, ``--maxval``)
   * - ``smlm_cluster``
     - `smlm-cluster`_
   * - ``groupprops``
     - `group-properties`_
   * - ``pc``
     - `pair-correlation`_, with ``--bin-size`` and ``--r-max`` (were
       ``--binsize``, ``--rmax``)
   * - ``localize --zc``, ``--mf``
     - ``--z-calibration``, ``--magnification-factor``
   * - ``undrift --fromfile``
     - ``--from-file``
   * - ``aim --intersectdist``, ``--roiradius``
     - ``--intersect-distance``, ``--roi-radius``
   * - ``dbscan --radius_z``
     - ``--radius-z``
   * - ``join --keepindex``
     - ``--keep-index``

.. _cmd-localize:

localize
--------
Localize identifies and fits single-molecule spots in a movie. Type
``picasso localize`` to open the GUI module, or ``picasso localize path args``
to run the analysis from the command line, where ``path`` is a movie file, a
folder or a file pattern.

.. code-block:: bash

   picasso localize path [args]

Finding out which arguments exist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``picasso localize -h`` prints every argument with its short and long name,
its type, its default value and a one-line description. Arguments that are not given keep
their defaults, so the shortest possible run is just
``picasso localize filename``.

The arguments fall into a few groups:

Spot identification
   Box side length, the identification method (``--identification-method``)
   with its threshold (the minimum net gradient, or the ``--wavelet-*``
   settings), and the pre-filters ``--temporal-median`` and
   ``--gaussian-filter``.
Fitting
   The fit method and the calibration files that some methods require (a
   spline PSF calibration for the spline fits, a magnification factor and a 3D
   calibration for the astigmatism fits).
Camera
   Baseline, sensitivity, gain and pixel size in nm.
What is analyzed
   Region of interest, frame bounds, and the flags that change how several
   movies or several regions are grouped (``--concat``,
   ``--regions-separately``).
Output
   Drift correction segmentation, a suffix for the output files, and whether
   the run is added to the local database.

The rest of this section explains only those options whose behavior needs
more than the one line that ``-h`` gives.

Batch process a folder
~~~~~~~~~~~~~~~~~~~~~~
To batch process a folder, simply type the folder name (or drag and drop it
into the console):

.. code-block:: bash

   picasso localize foldername

Picasso will analyze the folder and process all ``*.ome.tif`` files in it. If
the files have consecutive names (e.g., ``File.ome.tif``, ``File_1.ome.tif``,
``File_2.ome.tif``), they will be treated as one.

Drift correction
~~~~~~~~~~~~~~~~
Localize will automatically try to perform an RCC drift correction on the
dataset. As this will not always work with the default settings, after an
unsuccessful attempt the program will continue with the next file. If the
drift correction succeeds, another hdf5 file with the drift corrected locs
will be created.

Camera settings
~~~~~~~~~~~~~~~
Set the camera parameters (``--baseline``, ``--sensitivity`` and ``--gain``) to match your camera (see :ref:`localize-camera-config`). They convert
the camera counts to photons, so wrong values give wrong photon counts and
localization precisions. They also bias the maximum likelihood fits
(``-a mle``, the default, and the other ``mle`` methods), which assume
Poisson-distributed photon counts.

The camera pixel size (``-px``, ``--pixelsize``, in nm) converts the fitted
positions, spot widths and precisions from camera pixels to nm, the unit of
every localization file (see :doc:`files`). If it is not given, Picasso takes
the ``Pixelsize`` of the movie metadata, then the entry of the camera in the
camera configuration (:ref:`localize-camera-config`); if none of them records
it, ``picasso localize`` stops and asks for ``--pixelsize``. The value used is
printed and saved as ``Pixelsize`` in the metadata. ``spline-calibrate``
resolves it the same way; ``lateral-calibrate`` requires ``--pixelsize``.

Pre-filters
~~~~~~~~~~~
``--temporal-median``
   Subtracts a rolling per-pixel median background before spots are
   identified, which suppresses uneven background and static structures. It
   affects identification only. See :ref:`localize-temporal-median-filter`.
``--gaussian-filter``
   Smooths each frame with a Gaussian of the given standard deviation before
   spots are identified. It affects identification only. The minimum net gradient needs
   re-tuning when it is changed. It can be combined with
   ``--temporal-median``, which is applied first. See
   :ref:`localize-gaussian-filter`.

B-spline wavelet identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``--identification-method wavelet`` (``-im wavelet``) identifies spots by the
B-spline wavelet segmentation of `Izeddin et al., Optics Express, 2012 <https://doi.org/10.1364/OE.20.002081>`__, instead of by their net
gradient; ``--gradient`` is then ignored. See :ref:`localize-wavelet`.

``--wavelet-threshold``
   The threshold in units of the noise standard deviation (default 0.5).
``--wavelet-noise``
   How the noise is estimated: ``image-std``, the default, or ``w1-mad``,
   which is robust to dense spots and uneven background.
``--wavelet-min-area``
   The smallest region kept (default 4 camera pixels, i.e., an area on the
   detector grid).

The localizations have no ``net_gradient`` column. See
:ref:`localize-wavelet`. ``spline-calibrate`` and
``lateral-calibrate`` accept the same arguments for detecting beads:

.. code-block:: bash

   picasso localize movie.tif -b 7 -im wavelet --wavelet-threshold 1

Analyzing several movies as one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``--concat`` accumulates the movies found into one concatenated movie. Given a
folder, Picasso searches it and all of its sub-folders; given a pattern (e.g.,
``"experiment_folder/*/*.tif"``), it uses the matching files.

In both cases the files are ordered by folder and file name with numbers
compared numerically, so ``run_2`` comes before ``run_10``, and the full order
is printed before the analysis starts.

.. important::

   Check the printed order, since a wrong order is only noticeable afterwards.

Each entry is one whole movie: the continuation files of a split OME-TIFF
stack (``*_1.ome.tif``, ...) and the individual frames of a MicroManager
"separate image files" folder are read together with the movie they belong
to, so no frames are repeated.

Analyzing regions of one movie separately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``--regions-separately`` treats the ``--roi`` regions as channels imaged side
by side on one sensor and fits each of them on its own, writing
``<movie>_ref_locs.hdf5``, ``<movie>_ch1_locs.hdf5``, ... - one file per
region. Each file is in that region's own coordinates (the region's corner
subtracted from ``x`` and ``y``, and its size in the metadata), so the files
overlay each other when loaded as channels in Render.

The fit method, the minimum net gradient and the spline calibration may then
be given once per ``--roi``, in the same order as the regions, or once for all
of them:

.. code-block:: bash

   picasso localize movie.tif -b 7 --regions-separately --roi 0 0 256 256 --roi 0 256 256 512 -g 5000 -g 2000 -a lq -a mle

Separate movies need no flag: ``picasso localize`` already analyzes each file
on its own. See :ref:`localize-analyzing-each-channel`.

3D fitting
~~~~~~~~~~
If you select one of the astigmatism-based 3D algorithms (``lq-3d``,
``lq-gpu-3d`` or ``mle-3d``) you must supply both the magnification factor
(``-mf``) and the path to the 3D calibration file (``-zc``). If either is
omitted, the program will prompt you for it interactively. The spline fit
methods instead need a cubic-spline PSF calibration passed with
``--spline-calibration``.

Example
~~~~~~~
This example shows the batch process of a folder, with movie ome.tifs that are
supposed to be reconstructed and drift corrected with the ``lq`` algorithm and
a min. net gradient of 4000.

.. code-block:: bash

   picasso localize foldername -a lq -g 4000

.. _cmd-camera-calibrate:

camera-calibrate
----------------
Characterize an sCMOS camera: measure its per-pixel offset and readout
variance maps from a dark movie and, optionally, its per-pixel gain map from
movies at several illumination levels (see :ref:`localize-scmos-measuring`).
The calibration is saved as ``.hdf5``, together with a diagnostic plot of the
maps and their histograms, and is used with ``picasso localize -cm``.

.. code-block:: bash

   picasso camera-calibrate dark [-l light [-p power]] ... [-o output]

For example:

.. code-block:: bash

   picasso camera-calibrate dark.raw -l light_01.raw -p 0.1 -l light_02.raw -p 1 -l light_03.raw -p 10 -o mycam_scmos_calib.hdf5
   picasso localize movie.raw -cm mycam_scmos_calib.hdf5 -a mle

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Argument
     - Default
     - Description
   * - ``dark``
     -
     - Dark movie: frames recorded with no light on the sensor. Fewer than
       10,000 frames give a warning.
   * - ``-l``, ``--light`` (str)
     - none
     - Movie at a quasi-uniform illumination level, with the same camera
       settings as the dark movie. Repeat for several levels to measure the
       gain map; without it, only the offset and variance maps are measured.
   * - ``-p``, ``--power`` (float)
     - none
     - The illumination each light movie was recorded at (laser power,
       exposure time, ...), one per ``-l``, in the same order. It only labels
       the linearity plot; the gain fit does not use it.
   * - ``--power-unit`` (str)
     - ``mW``
     - The unit of ``-p``, for the plot axis.
   * - ``-o``, ``--output`` (str)
     - ``<dark>_scmos_calib.hdf5``
     - The output calibration file.

.. _cmd-camera-validate:

camera-validate
---------------
Check whether an sCMOS calibration from `camera-calibrate`_ still describes
the camera, using a short fresh dark movie (about 1,000 frames is plenty). The
mean per-pixel p-value is printed; outside 0.5 ± 0.1 the camera has drifted
and should be characterized again (see :ref:`localize-scmos-checking`).

.. code-block:: bash

   picasso camera-validate calibration movie

``calibration``
   The camera calibration (``.hdf5``).
``movie``
   The fresh dark movie.

.. _cmd-spline-calibrate:

spline-calibrate
----------------
Build a cubic-spline PSF calibration from a bead z-stack for the spline fit
methods of ``picasso localize`` (``--spline-calibration``), see
:ref:`localize-spline-building`. Pass one movie for a single-channel
calibration, several movies (one per channel) for a multichannel
calibration, or one movie with ``--split-fov`` when the channels are regions
of the same movie (see :ref:`localize-multichannel-spline-calibrating`). Next
to the calibration, a diagnostic plot and a bead gallery
(``<base>_beads.png``) are saved.

.. code-block:: bash

   picasso spline-calibrate files -s step [options]

For example, a z-stack with 20 nm steps:

.. code-block:: bash

   picasso spline-calibrate my_beads.tif -s 20

The beads are identified with the same arguments as in ``picasso localize``
(``-b``, ``-g``, ``-im`` and the ``--wavelet-*`` settings, see
`B-spline wavelet identification`_), and the camera pixel size is resolved as
described in `Camera settings`_.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Option
     - Default
     - Description
   * - ``-s``, ``--step`` (float)
     - required
     - The z step between consecutive stage positions (nm).
   * - ``-o``, ``--output`` (str)
     - ``<movie>_spline_calib.hdf5``
     - The output calibration file.
   * - ``-b``, ``--box-side-length`` (int)
     - ``13``
     - Box side length (camera pixels).
   * - ``-g``, ``--gradient`` (int)
     - ``5000``
     - Minimum net gradient for bead detection.
   * - ``-fps``, ``--frames-per-step`` (int)
     - ``1``
     - Number of frames acquired per z position (multi-FOV).
   * - ``-fo``, ``--frame-order``
     - ``fov``
     - ``fov`` or ``z``: the acquisition order when ``--frames-per-step`` is
       above 1.
   * - ``-zb``, ``--z-binning`` (int)
     - ``1``
     - Number of consecutive z steps averaged into one slice of the PSF
       model; e.g., 10 turns a 5 nm scan into 50 nm slices, which suppresses
       stage jitter that is correlated between neighboring steps.
   * - ``-m``, ``--model``
     - ``spline-3d``
     - ``spline-3d`` (recovers z) or ``spline-2d`` (single plane).
   * - ``-mf``, ``--magnification-factor`` (float)
     - ``0.79``
     - Scales the fitted z for the refractive-index mismatch. It is stored in
       the calibration and applied at fit time.
   * - ``-cz``, ``--correct-z-bias``
     - off
     - Set z = 0 at the axial intensity peak of the averaged PSF instead of
       the center of the stage scan. Only meaningful for a PSF with one
       well-defined focus (e.g., astigmatism).
   * - ``-rm``, ``--registration-model``
     - ``affine``
     - How the channels of a multichannel calibration are registered to the
       reference: ``translation``, ``affine``, ``projective``,
       ``polynomial2`` or ``polynomial3`` (see
       :ref:`localize-lateral-transform-models`). Ignored for one channel.
   * - ``-pr``, ``--photon-ratios`` (str)
     - none
     - Multichannel only: candidate per-channel photon ratios for
       ratiometric color assignment, hypotheses separated by ``;`` and
       channels by ``,`` (e.g., ``"0.7,0.3;0.4,0.6"``). Only relative values
       matter.
   * - ``-sf``, ``--split-fov`` (str)
     - none
     - Use rectangular regions of one movie as the channels, separated by
       ``;`` and given as ``y0,x0,y1,x1`` (camera pixels, all the same size),
       e.g., ``"0,0,512,256;0,256,512,512"``.
   * - ``-rf``, ``--reference`` (int)
     - ``0``
     - ``--split-fov`` only: the index of the reference region.
   * - ``-bl``, ``--baseline`` (float)
     - ``0``
     - Camera baseline.
   * - ``-se``, ``--sensitivity`` (float)
     - ``1``
     - Camera sensitivity.
   * - ``-ga``, ``--gain`` (int)
     - ``1``
     - Camera gain.
   * - ``-px``, ``--pixelsize`` (float)
     - from the metadata
     - Camera pixel size (nm), recorded in the calibration.

.. _cmd-lateral-calibrate:

lateral-calibrate
-----------------
Fit a lateral correction of ``x`` and ``y`` from two images of the same
in-focus beads (see :ref:`localize-lateral-calibrating`):

- ``--type astigmatism``: the ``reference`` without and the ``target`` with
  the cylindrical lens.
- ``--type chromatic``: the ``reference`` in the reference color channel and
  the ``target`` in the channel to be corrected.

The correction is appended to an existing 3D or spline calibration
(``--calibration``), which then applies it whenever it is used to fit, or
written to a standalone calibration (``--output``), which is loaded at fit
time with ``picasso localize -ac`` (see :ref:`localize-lateral-storage`).

.. code-block:: bash

   picasso lateral-calibrate reference target -px pixelsize (-c calibration | -o output) [options]

For example:

.. code-block:: bash

   picasso lateral-calibrate beads_without_lens.tif beads_with_lens.tif -px 130 -c beads_3d_calib.yaml
   picasso lateral-calibrate beads_red.tif beads_green.tif -t chromatic -px 130 -o chromatic.yaml -p chromatic_check.png

The beads are identified with the same arguments as in ``picasso localize``
(``-b``, ``-g``, ``-im`` and the ``--wavelet-*`` settings, see
`B-spline wavelet identification`_).

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Option
     - Default
     - Description
   * - ``-px``, ``--pixelsize`` (float)
     - required
     - Camera pixel size (nm). The correction is fitted on the bead images and
       stored in nm.
   * - ``-t``, ``--type``
     - ``astigmatism``
     - ``astigmatism`` or ``chromatic``: what the correction is for.
   * - ``-m``, ``--model``
     - ``affine``
     - ``translation``, ``affine``, ``projective``, ``polynomial2`` or
       ``polynomial3``, needing at least 1, 3, 4, 6 or 10 bead pairs (see
       :ref:`localize-lateral-transform-models`).
   * - ``-c``, ``--calibration`` (str)
     - none
     - An existing calibration (``.yaml`` or ``.hdf5``) to append the
       correction to.
   * - ``-o``, ``--output`` (str)
     - ``--calibration``
     - Where to write the calibration. Without ``--calibration``, a
       standalone lateral calibration is written here.
   * - ``-b``, ``--box-side-length`` (int)
     - ``7``
     - Box side length (camera pixels).
   * - ``-g``, ``--gradient`` (int)
     - ``5000``
     - Minimum net gradient for bead detection.
   * - ``-p``, ``--plot`` (str)
     - none
     - Save the diagnostic figure to this path.

.. _cmd-render:

render
------
Start the render module (GUI) or render from command line. With no arguments
the GUI opens; with a ``files`` path one or more localization files are
rendered to image files.

.. code-block:: bash

   picasso render [files] [options]

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Option
     - Default
     - Description
   * - ``-px``, ``--disp-px-size`` (float)
     - ``10.0``
     - The size of the rendered pixel in nm.
   * - ``-b``, ``--blur-method``
     - ``convolve``
     - One of ``none``, ``convolve``, ``gaussian``.
   * - ``-w``, ``--min-blur-width`` (float)
     - ``0.0``
     - Minimum blur width (nm) if blur is applied.
   * - ``--vmin`` (float)
     - ``0.0``
     - Minimum colormap level in range 0-100 or absolute value.
   * - ``--vmax`` (float)
     - ``20.0``
     - Maximum colormap level in range 0-100 or absolute value.
   * - ``--scaling``
     - ``yes``
     - ``yes`` or ``no``; if scaling, the colormap value is relative in the
       range 0-100.
   * - ``-c``, ``--cmap``
     -
     - The colormap to be applied: ``viridis``, ``inferno``, ``plasma``,
       ``magma``, ``hot`` or ``gray``.
   * - ``-s``, ``--silent``
     -
     - Do not open the rendered image file.

.. _cmd-filter:

filter
------
Start the filter module (GUI) or filter from the command line. With no
arguments the GUI opens; with a ``files`` path and one or more conditions on
the columns, written as in :ref:`Filter <filter-expression>`, the
localizations that satisfy them are kept. Several conditions are applied in
the given order. The result is saved as ``*_filter.hdf5`` with the conditions
in its metadata, so that Filter can :ref:`reapply them <filter-metadata>`.

.. code-block:: bash

   picasso filter [files conditions [conditions ...]]

For example:

.. code-block:: bash

   picasso filter "data/*_locs.hdf5" "100 < sx < 160" "lpx < 10"

Positional arguments:

``files``
   One or more hdf5 localization files (a unix style path pattern).
``conditions``
   One or more conditions; quote each one.

design
------
Start the design module (GUI).

.. code-block:: bash

   picasso design

simulate
--------
Start the simulation module (GUI).

.. code-block:: bash

   picasso simulate

average
-------
Start the 2D averaging module (GUI).

.. code-block:: bash

   picasso average

nanotron
--------
Start the nanoTRON module (GUI), see :doc:`nanotron`.

.. code-block:: bash

   picasso nanotron

server
------
Start the Picasso server (web browser GUI), see :doc:`server`. By default,
only this computer can open it, at ``http://localhost:8501``.

.. code-block:: bash

   picasso server [options]

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Option
     - Default
     - Description
   * - ``--network``
     -
     - Listen on all network interfaces, so that other computers can open
       the server. The server has no login, so anyone who can reach this
       computer can then use it, including the custom commands of the
       watcher, see :ref:`server-browser-app`.

.. _cmd-spinna:

spinna
------
Start the SPINNA module (GUI) or run batch analysis from command line. Without
``-p`` the GUI opens; passing ``-p path/to/parameters.csv`` runs batch analysis
from a parameters CSV file (see :ref:`spinna-batch-analysis` for the expected
CSV structure).

.. code-block:: bash

   picasso spinna -p path/to/parameters.csv

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-p``, ``--parameters`` (str)
     - .csv file containing the parameters for SPINNA batch analysis.
   * - ``-a``, ``--asynch``
     - Do not perform fitting asynchronously (multiprocessing).
   * - ``-b``, ``--bootstrap``
     - Perform bootstrapping.
   * - ``-v``, ``--verbose``
     - Display progress bar for each row.

.. _cmd-conversion:

.. _cmd-convert-units:

convert-units
-------------
Convert files written before Picasso 1.0, whose lengths are in camera pixels,
to the current format in nm (see :ref:`files-legacy`): localizations (and
their ``.yaml`` metadata), pick regions, drift ``.txt`` files, astigmatism and
lateral calibrations, SPINNA masks and nanoTRON models. Files that are in nm
already are skipped.

.. code-block:: bash

   picasso convert-units FILE [FILE ...] [--pixelsize PS] [--in-place]

.. list-table::
   :header-rows: 1

   * - Argument
     - Description
   * - ``files``
     - The files to convert.
   * - ``-px``, ``--pixelsize``
     - Camera pixel size in nm for files that do not record it (pick regions,
       drift files, calibrations and nanoTRON models always need it;
       localizations use the ``Pixelsize`` of their metadata). Default: from
       the metadata.
   * - ``--in-place``
     - Replace the original files. By default, the converted files are
       written next to the originals with ``_nm`` appended to their names
       (``movie_locs_nm.hdf5``).

csv2hdf
-------
Convert csv files (ThunderSTORM) to ``.hdf5``. ``-p/--pixelsize`` is the
camera pixel size in nm, recorded in the metadata (``Pixelsize``) and used to
derive the field of view (``Width``, ``Height``) from the coordinates; it is
required. The coordinates stay in nm, as in ThunderSTORM.

.. code-block:: bash

   picasso csv2hdf filepath -p pixelsize

Note that the following columns need to be present:

2D files
   ``frame, x_nm, y_nm, sigma_nm, intensity_photon, offset_photon,
   uncertainty_xy_nm``
3D files
   ``frame, x_nm, y_nm, z_nm, sigma1_nm, sigma2_nm, intensity_photon,
   offset_photon, uncertainty_xy_nm``

hdf2csv
-------
Convert hdf5 files to ``.csv`` files (keeps column names).

.. code-block:: bash

   picasso hdf2csv files

hdf2ts
------
Convert hdf5 files to ThunderSTORM ``.csv`` files (adapts column names).

.. code-block:: bash

   picasso hdf2ts files

hdf2imagej
----------
Convert hdf5 files to ImageJ ``.txt`` files.

.. code-block:: bash

   picasso hdf2imagej files

hdf2nis
-------
Convert hdf5 files to NIS Elements ``.txt`` files.

.. code-block:: bash

   picasso hdf2nis files

hdf2chimera
-----------
Convert hdf5 files to Chimera ``.xyz`` files.

.. code-block:: bash

   picasso hdf2chimera files

hdf2visp
--------
Convert hdf5 files to ViSP ``.3d`` files.

.. code-block:: bash

   picasso hdf2visp files

smap2hdf
--------
Convert `SMAP <https://github.com/jries/SMAP>`_ ``_sml.mat`` files to
``.hdf5``. ``-p/--pixelsize`` is the camera pixel size in nm, recorded in
the metadata (``Pixelsize``) and used to derive the field of view from the
coordinates; it is required. The coordinates stay in nm, as in SMAP.

.. code-block:: bash

   picasso smap2hdf filepath -p pixelsize

Reads single-file MATLAB ``-v7`` and ``-v7.3`` saves; split ``_sml_p*.mat``
parts files are not supported (re-save them in SMAP as a single file).

hdf2smap
--------
Convert hdf5 files to SMAP ``_sml.mat`` files. The output is named
``<file>_sml.mat`` (the ``_sml`` suffix is required for SMAP to recognize the
file) and can be loaded directly in SMAP via ``File > Load``.

.. code-block:: bash

   picasso hdf2smap files

pyme2hdf
--------
Convert PyME (`python-microscopy <https://python-microscopy.org>`_) ``.h5r``
(raw fit results) and ``.hdf`` (saved in PYMEVisualize) files to ``.hdf5``.
The camera pixel size is read from the file; ``-p/--pixelsize`` in nm is used
for files that do not store it. A file with several channels (PyME's ``probe``
column) is saved as one file per channel, ``<file>_probe<k>_locs.hdf5``. See
:ref:`render-export-pyme` for how the columns are converted.

.. code-block:: bash

   picasso pyme2hdf filepath [-p pixelsize]

hdf2pyme
--------
Convert hdf5 files to PyME ``.hdf`` files, which can be opened in
PYMEVisualize. The output is named ``<file>.hdf`` (PyME does not recognize the
``.hdf5`` extension).

.. code-block:: bash

   picasso hdf2pyme files

join
----
Combine two hdf5 localization files. A new joined file will be created.

.. code-block:: bash

   picasso join file1 file2

.. warning::

   The frame information of consecutive files is reindexed, i.e., frame 1 now
   can contain localizations from file 1 and file 2. Therefore, do not perform
   kinetic analysis and drift correction on joined files. Pass
   ``-k/--keep-index`` to keep the original frame numbers instead of
   reindexing.

Columns that not all files have (e.g., ``z`` when joining 2D and 3D files, or
``net_gradient`` when only some were identified by their net gradient) are
dropped with a warning, so that no localizations are lost.

.. _cmd-stitch:

stitch
------
Stitch hdf5 localization files of tiles recorded at different stage
positions into one file, see :ref:`render-stitch-tiles`. The tiles are placed
by the stage positions in the Micro-Manager metadata and aligned by matching
their overlaps. Files at the same stage position (e.g., the targets of an
Exchange-PAINT experiment) are aligned together, and the tiles of each target
(the n-th file of each position, in the order given) are saved as one file,
``<first file>_stitched.hdf5``. The tile positions and the offsets between
overlapping tiles are printed.

The stage positions are required. Files without them (e.g., not acquired with
Micro-Manager) can be stitched in Render after placing the tiles with the
Move tool, see :ref:`render-stitch-without-stage-positions`, or with
``stitch.stitch`` and its ``positions`` argument in Python.

.. code-block:: bash

   picasso stitch files [-o output] [-r search_radius] [--orientation auto] [--2d] [--keep-overlaps] [--keep-frames]

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``-o``, ``--output`` (str)
     - ``<first file>_stitched.hdf5``
     - The output file. With several targets, ``_target<n>`` is appended.
   * - ``-r``, ``--search-radius`` (float)
     - ``10``
     - How far the true offset of two tiles may be from the one given by the
       stage positions (µm).
   * - ``--orientation`` (str)
     - ``auto``
     - How the stage axes run in the image, i.e., where a tile at a larger
       stage position lies: e.g., ``"-x, -y"`` if a larger stage X is to the
       left and a larger stage Y above (the image y axis points down). All
       eight are listed in :ref:`render-stitch-stage-orientation`. ``auto``
       finds the one whose overlaps match best.
   * - ``--2d``
     - off
     - Do not correct the z offsets of the tiles.
   * - ``--keep-overlaps``
     - off
     - Keep the localizations of all tiles in the overlaps instead of cutting
       them at the seams.
   * - ``--keep-frames``
     - off
     - Do not offset the frames of consecutive tiles.

link
----
Link localizations in consecutive frames.

.. code-block:: bash

   picasso link files [-d distance] [-t tolerance]

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``-d``, ``--distance`` (float)
     - ``10``
     - Maximum distance between localizations to consider them the same
       binding event (nm).
   * - ``-t``, ``--tolerance`` (int)
     - ``1``
     - Maximum dark time between localizations to still consider them the
       same binding event.

.. _cmd-track:

track
-----
Link localizations into tracks by nearest-neighbor tracking (single-particle
tracking, see :ref:`render-spt-track`). Takes one or more hdf5 localization
files (unix style path patterns). Each is saved as
``<file name>_tracks.hdf5``, with the track id of every localization in the
``group`` column; a previous ``group`` column is kept as ``group_input``.
Linked localizations (see `link`_) are skipped.

.. code-block:: bash

   picasso track files [-d max-jump] [-g max-gap] [-m min-length] [-t frame-time]

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``-d``, ``--max-jump`` (float)
     - ``500``
     - Maximum distance between consecutive localizations of a track (nm).
   * - ``-g``, ``--max-gap`` (int)
     - ``1``
     - Maximum number of frames a track may skip.
   * - ``-m``, ``--min-length`` (int)
     - ``2``
     - Minimum number of localizations of a track.
   * - ``-t``, ``--frame-time`` (float)
     - none
     - Time between two consecutive frames (ms), stored in the metadata for
       ``picasso diffusion``.

.. _cmd-diffusion:

diffusion
---------
Diffusion analysis of tracked localizations (see `track`_ and
:ref:`render-spt-analysis`): the mean squared displacement, the fits of the
jump distances and a summary of every track. For each file, saves
``<file name>_track_stats.csv``, ``<file name>_msd.csv`` and
``<file name>_diffusion.yaml``, and prints the diffusion coefficients.

.. code-block:: bash

   picasso diffusion files [-t frame-time] [-l max-lag] [-n n-fit] [-k populations] [-s sigma] [--truncate] [--plot]

For example, fits of one and two populations with the localization error set
to the median localization precision, saving the plots:

.. code-block:: bash

   picasso diffusion movie_locs_tracks.hdf5 -k 1 2 -s lp --plot

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``-t``, ``--frame-time`` (float)
     - from the metadata
     - Time between two consecutive frames (ms).
   * - ``-l``, ``--max-lag`` (int)
     - ``10``
     - Largest time lag of the MSD (frames).
   * - ``-n``, ``--n-fit`` (int)
     - ``4``
     - Number of time lags fitted in the MSD analysis.
   * - ``-k``, ``--populations`` (int)
     - ``1 2 3``
     - Number(s) of diffusing populations fitted to the jump distances.
   * - ``-s``, ``--sigma`` (float or ``lp``)
     - ``0``
     - Localization error for the jump distance fit (nm), or ``lp`` for the
       median localization precision. 0 gives apparent diffusion
       coefficients.
   * - ``--truncate``
     - off
     - Correct the jump distance fit for the maximum jump distance of the
       tracking.
   * - ``--plot``
     - off
     - Save the plots of the MSD, the jump distances and the track lengths
       as ``.png`` files.

cluster-filter
--------------
Filter localizations by properties of their clusters.

.. code-block:: bash

   picasso cluster-filter files -c cluster_file -p parameter --min-value min --max-value max

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-c``, ``--cluster-file`` (str)
     - The cluster centers, e.g., ``_cluster_centers.hdf5``.
   * - ``-p``, ``--parameter`` (str)
     - Parameter to be filtered.
   * - ``--min-value`` (float)
     - Lower boundary.
   * - ``--max-value`` (float)
     - Upper boundary.

undrift
-------
Correct localization coordinates for drift with RCC.

.. code-block:: bash

   picasso undrift files [-s segmentation] [-f fromfile] [-d]

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``-s``, ``--segmentation`` (float)
     - ``1000``
     - The number of frames to be combined for one temporal segment.
   * - ``-f``, ``--from-file`` (str)
     -
     - Apply drift from specified file instead of computing it.
   * - ``-d``, ``--display``
     -
     - Display estimated drift.

aim
---
Correct localization coordinates for drift with AIM.

.. code-block:: bash

   picasso aim files [-s segmentation] [-i intersectdist] [-r roiradius]

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``-s``, ``--segmentation`` (float)
     - ``100``
     - The number of frames to be combined for one temporal segment.
   * - ``-i``, ``--intersect-distance`` (float)
     - ``20``
     - Max. distance (nm) between localizations in consecutive segments to
       be considered as intersecting.
   * - ``-r``, ``--roi-radius`` (float)
     - ``60``
     - Max. drift (nm) between two consecutive segments.

undrift-fiducials
-----------------
Correct localization coordinates for drift using fiducial markers
(automatically picked). Takes one or more hdf5 localization files as
positional arguments.

.. code-block:: bash

   picasso undrift-fiducials files

density
-------
Compute the local density of localizations. Takes positional ``files`` and
``radius`` (float): the maximal distance (nm) between two localizations to be
considered local.

.. code-block:: bash

   picasso density files radius

dbscan
------
Cluster localizations with the DBSCAN clustering algorithm.

.. code-block:: bash

   picasso dbscan files radius density

Positional arguments:

``files``
   One or more hdf5 localization files.
``radius`` (float)
   Maximal distance (nm) between two localizations to be considered local.
``density`` (int)
   Minimum local density for localizations to be assigned to a cluster.

hdbscan
-------
Cluster localizations with the HDBSCAN clustering algorithm.

.. code-block:: bash

   picasso hdbscan files min_cluster min_samples

Positional arguments:

``files``
   One or more hdf5 localization files.
``min_cluster`` (int)
   Smallest size grouping that is considered a cluster.
``min_samples`` (int)
   The higher, the more points are considered noise.

smlm-cluster
------------
Cluster localizations with the custom SMLM clustering algorithm. The algorithm
finds localizations with the most neighbors within a specified radius and
finds clusters based on such "local maxima".

.. code-block:: bash

   picasso smlm-cluster files radius min_locs [basic_fa] [radius_z]

Positional arguments:

``files``
   One or more hdf5 localization files.
``radius`` (float)
   Clustering radius (nm).
``min_locs`` (int)
   Minimum number of localizations in a cluster.
``basic_fa`` (bool)
   Whether to perform basic frame analysis (sticking event removal).
``radius_z`` (float)
   Clustering radius (nm) in axial direction (MUST BE SET FOR 3D!).

g5m
---
Run Gaussian Mixture Modeling with Modifications (G5M) for Molecular Mapping
on clustered localizations. For details see
https://doi.org/10.1038/s41467-026-70198-5.

The positional ``files`` argument is a unix-style path to one or more
clustered ``.hdf5`` files, or a folder in which all ``.hdf5`` files will be
analyzed. If omitted, the GUI is launched.

.. code-block:: bash

   picasso g5m files [options]

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``-ml``, ``--min-locs`` (int)
     - ``10``
     - Min. number of locs per molecule.
   * - ``-lph``, ``--loc-prec-handle`` (str)
     - ``local``
     - Loc. precision handle, either ``local`` or ``abs``.
   * - ``--min-sigma`` (float)
     - ``0.8``
     - Minimum sigma factor/value.
   * - ``--max-sigma`` (float)
     - ``1.5``
     - Maximum sigma factor/value.
   * - ``--max-rounds`` (int)
     - ``3``
     - Max. rounds without BIC improvement to terminate.
   * - ``--bootstrap-sem``
     -
     - Bootstrap to estimate SEM of molecule positions.
   * - ``-c``, ``--calibration`` (str)
     - ``''``
     - Path to calibration file (3D only).
   * - ``--covariance-type`` (str)
     - ``auto``
     - Shape of the fitted molecules: ``auto``, ``spherical``, ``diagonal``
       or ``rotated``. ``auto`` picks ``rotated`` for 3D astigmatism locs
       carrying an ``angle`` column, ``diagonal`` for 3D localizations
       without the ``angle`` column and ``spherical`` for 2D data.
   * - ``-p``, ``--postprocess``
     -
     - Do not postprocess results to remove sticking events and low-quality
       fits.
   * - ``--max-locs`` (int)
     - ``100000``
     - Maximum number of localizations to process per cluster; useful for
       excluding fiducials.
   * - ``-a``, ``--asynch``
     -
     - Do not perform fitting asynchronously (multiprocessing).

dark
----
Compute the dark time for grouped localizations.

.. code-block:: bash

   picasso dark files

align
-----
Align one localization file to another via RCC, 2D only (two or more files). The
RCC is repeated until the shifts converge, as in Render. Pass ``-d/--display``
to plot the mean shift per iteration. All localizations are kept by
default; pass ``-r/--remove-non-overlapping`` to remove the localizations
outside the region where all aligned files overlap (see
:ref:`render-align-non-overlapping`).

.. code-block:: bash

   picasso align file1 file2 [...]
   picasso align file1 file2 --remove-non-overlapping

group-properties
----------------
Calculate the properties of localization groups.

.. code-block:: bash

   picasso group-properties files

pair-correlation
----------------
Calculate the pair-correlation of localizations.

.. code-block:: bash

   picasso pair-correlation files [-b bin_size] [-r r_max]

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``-b``, ``--bin-size`` (float)
     - ``10``
     - The bin size (nm).
   * - ``-r``, ``--r-max`` (float)
     - ``1000``
     - The maximum distance (nm) to calculate the pair-correlation.

nearest-neighbor
----------------
Calculate the nearest neighbor within a clustered dataset.

.. code-block:: bash

   picasso nearest-neighbor files

cbc
---
Coordinate-based colocalization (CBC) of localizations with other
localization files (Malkusch, et al., Histochem Cell Biol, 2012, DOI:
`10.1007/s00418-011-0880-5 <https://doi.org/10.1007/s00418-011-0880-5>`__).
Takes two or more hdf5 localization files (unix style path patterns). The
first file is scored against all others; with ``-a``, every file is scored
against every other file, i.e., both directions of every pair (also for two
files). The CBC is not symmetric, so both directions of a pair differ. Each
scored file is saved as ``<file name>_cbc.hdf5`` with a column
``cbc_<name>`` per partner, where the names are the file names or those given
with ``-n``. z is used if all files contain it.
The files should be linked first (see `link`_); a warning lists the ones
that are not.

.. code-block:: bash

   picasso cbc files [-r r_max] [-s steps] [-u undefined] [-a] [--2d] [-n names]

For example, both directions for two files with short names:

.. code-block:: bash

   picasso cbc long_name_bassoon_locs.hdf5 long_name_psd95_locs.hdf5 -a -n Bassoon PSD95

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``-r``, ``--r-max`` (float)
     - ``200``
     - The maximum radius (nm).
   * - ``-s``, ``--steps`` (int)
     - ``100``
     - The number of radii up to the maximum radius.
   * - ``-u``, ``--undefined`` (float or ``remove``)
     - ``0``
     - The CBC given to localizations without neighbors within the maximum
       radius, between -1 and 1, or ``remove`` to remove them (if they
       have no neighbors for at least one partner).
   * - ``-a``, ``--all-pairs``
     - off
     - Score every file against every other file (both directions).
   * - ``-n``, ``--names`` (str)
     - file names
     - Short names of the files in the order given, used for the column
       names ``cbc_<name>``.
   * - ``--2d``
     - off
     - Ignore z and compute the CBC in 2D.

ripley-h
--------
Ripley's H function of localizations (Ripley, J R Stat Soc B, 1977, DOI:
`10.1111/j.2517-6161.1977.tb01615.x
<https://doi.org/10.1111/j.2517-6161.1977.tb01615.x>`__): whether the
molecules are clustered (H > 0), randomly distributed (H = 0) or dispersed
(H < 0), and on which length scales. Computed in 2D or, with ``--3d``, in 3D
in the picks given with ``-p`` (averaged over the picks) or in the whole field
of view, with Ripley's
isotropic edge correction and an envelope of complete spatial randomness
(CSR) from simulations. The files should contain molecules, i.e., one
localization per molecule, such as those found by `g5m`_ or the cluster
centers saved by `dbscan`_, `hdbscan`_ or `smlm-cluster`_; a warning lists the
files without molecules (no ``n_events`` or ``std_frame`` column). See
:ref:`render-ripley-h` for details.

Saves next to each file ``<file name>_ripley_h.csv`` (the functions by
radius: mean, CSR envelope and each region), ``<file
name>_ripley_h_regions.csv`` (number of localizations, area, density, maximum
of H and its radius in each region) and the plot ``<file
name>_ripley_h.png``.

.. code-block:: bash

   picasso ripley-h files [-p picks] [-r r_max] [-s steps] [-n simulations] [--no-edge-correction] [--3d] [--z-range zmin zmax] [--observed-z] [--partner file] [--cross-null shift|labels]

For example, in the picks saved in Render, and in 3D between z = -300 and 300
nm:

.. code-block:: bash

   picasso ripley-h molecules.hdf5 -p picks.yaml -r 300
   picasso ripley-h molecules.hdf5 -p picks.yaml -r 200 --3d --z-range -300 300

and between two channels, with random shifts of the second one:

.. code-block:: bash

   picasso ripley-h protein_a_molecules.hdf5 --partner protein_b_molecules.hdf5 -r 300

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``-p``, ``--picks`` (str)
     - whole field of view
     - Picks (.yaml) saved in Render; the regions of interest in which H is
       computed.
   * - ``-r``, ``--r-max`` (float)
     - ``500``
     - The maximum radius (nm).
   * - ``-s``, ``--steps`` (int)
     - ``100``
     - The number of radii up to the maximum radius.
   * - ``-n``, ``--simulations`` (int)
     - ``99``
     - The number of CSR simulations for the envelope, ``0`` to skip them.
   * - ``--no-edge-correction``
     - off
     - Do not apply Ripley's isotropic edge correction.
   * - ``--3d``
     - off
     - Compute H in 3D (x, y and z); requires z. See
       :ref:`render-ripley-h-3d`.
   * - ``--z-range`` (float float)
     - z of each region
     - The z range (nm) over which the regions extend in 3D. By default, each
       region extends over the z range of its localizations.
   * - ``--observed-z``
     - off
     - In 3D, give the simulated points the observed z values (shuffled)
       instead of spreading them uniformly over the z range.
   * - ``--partner`` (str)
     - none
     - A localization file of a second channel (aligned). The cross function
       of each file with it is computed and saved as
       ``<file name>_ripley_h_cross.csv``, etc. See
       :ref:`render-cross-functions`.
   * - ``--cross-null`` (``shift`` or ``labels``)
     - ``shift``
     - With ``--partner``, how the envelope is simulated: random shifts of
       the partner (independent channels; rectangular or circular regions
       only) or random labels (both channels follow the same distribution).

rdf
---
Radial distribution function g(r) of localizations: the density of neighbors
at the distance r from a molecule, relative to the mean density (1 for
complete spatial randomness, above 1 at the distances where molecules are more
likely to be found than at random, below 1 where they are depleted). Computed
like `ripley-h`_, with the same options and outputs (``<file name>_rdf.csv``,
``<file name>_rdf_regions.csv`` and ``<file name>_rdf.png``), except that the
radius steps are rings of ``rmax / steps`` and the default number of steps is
50. See :ref:`render-rdf` for details.

.. code-block:: bash

   picasso rdf files [-p picks] [-r r_max] [-s steps] [-n simulations] [--no-edge-correction] [--3d] [--z-range zmin zmax] [--observed-z] [--partner file] [--cross-null shift|labels]

For example, in the picks saved in Render with rings of 5 nm:

.. code-block:: bash

   picasso rdf molecules.hdf5 -p picks.yaml -r 300 -s 60

nnd
---
Distributions of the first nearest neighbor distances (NNDs) within and
between channels, compared with random simulations (complete spatial
randomness within channels; random shifts or random labels between channels).
Takes the localization files of the channels (aligned molecules, unix style
path patterns), in the picks given with ``-p`` or in the whole field of view.
Saves next to the first file ``<file name>_nnd.csv`` (the histograms of every
ordered pair of channels), ``<file name>_nnd_summary.csv`` (number of
molecules and median NND, measured and simulated, of every pair) and the plot
``<file name>_nnd.png``, and prints the summary. See
:ref:`render-nnd-distributions` for details.

.. code-block:: bash

   picasso nnd files [--names names] [-p picks] [-r r_max] [-s steps] [-n simulations] [--3d] [--z-range zmin zmax] [--observed-z] [--cross-null shift|labels]

For example, two channels with short names:

.. code-block:: bash

   picasso nnd protein_a_molecules.hdf5 protein_b_molecules.hdf5 --names A B

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``--names`` (str)
     - file names
     - Names of the channels in the order of the files.
   * - ``-p``, ``--picks`` (str)
     - whole field of view
     - Picks (.yaml) saved in Render; the regions of interest.
   * - ``-r``, ``--r-max`` (float)
     - ``200``
     - The largest distance of the histograms (nm).
   * - ``-s``, ``--steps`` (int)
     - ``40``
     - The number of histogram bins up to the largest distance.
   * - ``-n``, ``--simulations`` (int)
     - ``99``
     - The number of random simulations, ``0`` to skip them.
   * - ``--3d``
     - off
     - Compute the distances in 3D (x, y and z); requires z.
   * - ``--z-range`` (float float)
     - z of each region
     - The z range (nm) over which the regions extend in 3D.
   * - ``--observed-z``
     - off
     - In 3D, give the simulated points within a channel the observed z
       values (shuffled) instead of spreading them uniformly.
   * - ``--cross-null`` (``shift`` or ``labels``)
     - ``shift``
     - How pairs of channels are simulated: random shifts of the second
       channel (rectangular or circular regions only) or random labels.

.. _cmd-plugins:

plugins
-------
Manage Picasso plugins without opening a GUI.

.. code-block:: bash

   picasso plugins list
   picasso plugins enable <file.py>
   picasso plugins disable <file.py>
   picasso plugins install <id>
   picasso plugins update <id>
   picasso plugins uninstall <id>
   picasso plugins path

``list``
   Shows every plugin file in ``~/.picasso/plugins``, whether it is enabled,
   and what each enabled one contributes (a GUI menu entry, command line
   commands, API names).
``enable <file.py>``, ``disable <file.py>``
   Control whether a file is loaded and run at all.
``install``, ``update``, ``uninstall``
   Work against the online registry with the same hash verification as the
   plugin browser.
``path``
   Prints the plugins folder.

Plugins can add their own commands here: an enabled plugin that defines
``register_cli`` contributes subcommands that appear in ``picasso -h`` and
behave like the built-in ones. See :ref:`plugins-for-developers` for how to
write one, and :ref:`plugins-without-gui` for more on managing plugins from
the command line.
