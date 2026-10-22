.. _migration:

Migrating to Picasso 1.0
========================

Picasso 1.0 is the first version with a stability promise (see
:doc:`versioning`), and it cleans up a lot at once to keep that promise
manageable: every length is in nm instead of camera pixels, the command line
has one naming style, and the Python API has one way to pass identification
settings and to report progress. This page lists what changes for you, coming
from Picasso 0.11, and maps each old setting, command and function to its
replacement. The `release notes
<https://github.com/jungmannlab/picasso/blob/master/changelog.md>`__ list
every change, including the new features.

What applies to you:

- :ref:`Your files <migration-files>` keep working; there is nothing to do.
- :ref:`In the Picasso windows <migration-gui>`, a few settings are renamed,
  and coordinates you type are in nm.
- :ref:`On the command line <migration-cli>`, the old command names keep
  working, but options that take a distance now take nm.
- :ref:`Some results change <migration-results>` with the same settings,
  e.g., where a default was in camera pixels.
- :ref:`Python scripts <migration-python>` need the most changes: lengths in
  nm, circular picks sized by their diameter, new arguments for
  identification and progress, and some removed and renamed functions.
- :ref:`Plugins <migration-plugins>` keep their hooks; some attributes of the
  Render window changed.

.. _migration-files:

Your files
----------

**Every file written by an earlier version opens in Picasso 1.0 as before.**
Picasso converts its lengths from camera pixels to nm as it reads the file,
with the ``Pixelsize`` of its metadata (Localize records it; if a file lacks
it, the windows ask for it), and never changes the file on disk. This holds
for localizations, pick regions, drift files, calibrations, SPINNA masks,
nanoTRON models and Render's field of view and view settings. See
:ref:`files-legacy` for the details.

Three things to know when you work with others or with other software:

- **Files written by Picasso 1.0 are in nm**, and earlier versions read them
  as camera pixels. **Whoever opens your new files needs Picasso 1.0 or later.**
- **Your own code that reads Picasso files directly** (e.g., with ``h5py``,
  pandas or MATLAB) gets ``x``, ``y``, ``lpx``, ``lpy``, ``sx`` and ``sy`` in
  nm from files written by 1.0, so it must no longer multiply them by the
  pixel size. A file written by 1.0 has the attribute ``picasso_format`` at
  the root of the HDF5 file (see :ref:`files-format-versions`); a file
  without it is in camera pixels. The same holds for the exports:
  ``picasso hdf2csv`` and the ``.txt`` for ImageJ (``picasso hdf2imagej`` and
  Render) write nm.
- **Files written by other software** for Picasso must carry the format
  marker to be read as nm, see the note in :ref:`files-legacy`.

Converting your files on disk is optional: it saves the question for the pixel
size and makes the files readable as nm by other software. Use
``File > Convert files to nm...`` in Render (see
:ref:`render-convert-files`), ``picasso convert-units`` (see
:ref:`cmd-convert-units`) or :func:`picasso.legacy.convert_files`. Astigmatism
(3D) and lateral calibrations do not record the pixel size: Picasso converts
them with the pixel size of the movie or localizations they are used with,
which assumes they were recorded with the same camera. To convert one with
another pixel size, convert it once, e.g.:

.. code-block:: bash

   picasso convert-units astig_3d_calib.yaml --pixelsize 130

.. _migration-gui:

In the Picasso windows
----------------------

Most dialogs took nm already in 0.11 (linking, DBSCAN, HDBSCAN, the SMLM
clusterer, AIM, RESI, pick sizes, the scale bar, ...) and are unchanged. What
changed:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Picasso 0.11
     - Picasso 1.0
   * - Render, *Display settings* > *Zoom* (screen pixels per camera pixel)
     - *Display width (µm)*, the width of the region shown in the window.
   * - Render, *Display settings* > *Camera pixel size (nm)*
     - Removed: each channel's pixel size is the ``Pixelsize`` of its
       metadata. Channels with different pixel sizes are shown together, with
       a warning instead of a question for one pixel size.
   * - Coordinates typed or shown in camera pixels: *Change field of view*,
       *Export view manually*, the view in *Show info*, *Apply expression*
       (``x += 1`` moved by one camera pixel; ``spiral R N``), the ranges in
       Filter
     - The same in nm: ``x += 100`` moves the localizations by 100 nm. The
       field of view, picks and screenshot files are saved in nm.
   * - Render, ``View > Files...``: *Rel. intensity* of each channel
     - *Contrast (% of global)*, a low and a high limit per channel. A
       relative intensity ``r`` of 1 or more corresponds to a high limit of
       ``100 / r`` %, see :ref:`Channel contrast <render-channel-contrast>`.
   * - ``View > Update rotation window`` (Ctrl+Shift+R)
     - ``View > 3D view``, same shortcut, see :doc:`render/3d`.
   * - ``Tools > Fast rendering...``
     - Removed: GPU rendering and the faster CPU rendering make it
       unnecessary, see :doc:`render/performance`.
   * - 3D window, ``View > Show/hide rotation`` (the axes icon)
     - Removed; the reference grid shows the axes, see
       :ref:`render-grid`.
   * - ``Tools > Filter picks by number of localizations...``
     - ``Tools > Filter picks by count...``
   * - *Left*, *Right*, *Up*, *Down*, *Zoom in* and *Zoom out* in the View
       menus of Render, its 3D window and Localize
     - Removed from the menus; their keyboard shortcuts still work.
   * - Localize, *Parameters* > *Quantum efficiency*
     - Removed: it had no effect on the photon counts. A
       ``Quantum Efficiency`` entry in the camera configuration is ignored.
   * - Picasso: Average3 (``picasso average3``)
     - Removed.
   * - Picasso Server could be opened from other computers
     - Only from the computer it runs on; start it with
       ``picasso server --network`` to allow other computers, see
       :ref:`server-browser-app`.

.. _migration-cli:

On the command line
-------------------

**Renamed commands and options keep working** until Picasso 2.0 and print the
new name: ``undrift_fiducials``, ``nneighbor``, ``clusterfilter``,
``smlm_cluster``, ``groupprops`` and ``pc`` are now ``undrift-fiducials``,
``nearest-neighbor``, ``cluster-filter``, ``smlm-cluster``,
``group-properties`` and ``pair-correlation``, and options such as
``--rmax`` are now ``--r-max``. The short forms (``-r``, ``-zc``, ...) are
unchanged. :ref:`cmd-renamed` lists them all.

**Lengths are in nm.** The options that took camera pixels:

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - Command and option
     - Picasso 0.11
     - Picasso 1.0
   * - ``link -d/--distance``
     - 1 camera pixel (default)
     - 10 nm (default)
   * - ``aim -i/--intersect-distance``
     - 0.154 camera pixels (default)
     - 20 nm (default)
   * - ``aim -r/--roi-radius``
     - 0.46 camera pixels (default)
     - 60 nm (default)
   * - ``pair-correlation -b/--bin-size``
     - 0.1 camera pixels (default)
     - 10 nm (default)
   * - ``pair-correlation -r/--r-max``
     - 10 camera pixels (default)
     - 1000 nm (default)
   * - ``density`` ``radius``
     - camera pixels
     - nm
   * - ``dbscan`` ``radius`` and ``--radius-z``
     - camera pixels
     - nm
   * - ``smlm-cluster`` ``radius`` and ``radius_z``
     - camera pixels
     - nm
   * - ``render -w/--min-blur-width``
     - camera pixels
     - nm
   * - ``cluster-filter --min-value/--max-value`` of a length (e.g.,
       ``std_x``)
     - camera pixels
     - nm
   * - ``g5m --min-sigma/--max-sigma`` with ``-lph abs``
     - camera pixels
     - nm

For example, with a camera pixel size of 130 nm:

.. code-block:: bash
   :caption: Picasso 0.11

   picasso link locs.hdf5 -d 0.5
   picasso dbscan locs.hdf5 0.1 4 130
   picasso pc locs.hdf5 --binsize 0.1 --rmax 10

.. code-block:: bash
   :caption: Picasso 1.0

   picasso link locs.hdf5 -d 65
   picasso dbscan locs.hdf5 13 4
   picasso pair-correlation locs.hdf5 --bin-size 13 --r-max 1300

**Removed:**

- the positional ``pixelsize`` of ``dbscan``, ``hdbscan`` and
  ``smlm-cluster``: old command lines with it stop with an error
  (``z`` was in nm already, and the radii are in nm now);
- the commands ``average3``, ``cluster_combine`` and ``cluster_combine_dist``;
- the cluster tables written by DBSCAN and HDBSCAN before Picasso 0.5
  (``_clusters.hdf5`` with a ``clusters`` dataset): ``nearest-neighbor``
  and ``cluster-filter`` read the cluster centers
  (``_cluster_centers.hdf5``), and ``link`` no longer adds linking
  statistics to such a file.

**The pixel size** of ``localize -px/--pixelsize`` and
``spline-calibrate -px/--pixelsize`` comes from the movie's metadata or the
camera configuration when not given (it was 130 nm), and the command stops if
neither has one. ``lateral-calibrate`` requires ``-px/--pixelsize``.
``localize --qe`` is accepted and ignored.

**Other changes:** ``align`` repeats the RCC alignment until the shifts
converge, as Render does. ``undrift --from-file`` also corrects z if the drift
file has it. ``picasso server`` needs the ``server`` extra
when Picasso is installed with pip (see :ref:`migration-installing`) and
accepts connections from other computers only with ``--network``.

**The metadata the commands write** comes from the Python functions they run
(see :ref:`migration-python-locs-info`), so a command records the same as a
script, and some keys changed. Scripts that read these keys from the
metadata need the new ones:

- ``link``: ``Max. distance (nm)`` and ``Max. transient dark frames``
  (were ``Maximum Distance`` and ``Maximum Transient Dark Time``).
- ``undrift``: ``Segmentation`` (RCC) as before, and no ``Drift X`` and
  ``Drift Y`` (the drift is saved in the drift file). With ``--from-file``,
  the step is ``Apply drift`` with ``From File``.
- ``undrift-fiducials``: two entries, one for the RCC before picking the
  fiducials (``Segmentation``, was ``pre-RCC segmentation``) and one for the
  fiducials (``Number of picks``, ``Pick shape`` and ``Pick diameter (nm)``,
  was ``Pick radius (nm)``).
- ``density``: ``Radius (nm)`` (was ``Radius``).
- ``align``: the shift subtracted from each file (``Shift x (nm)``,
  ``Shift y (nm)``) and the method, besides ``Files``.
- ``dbscan``, ``hdbscan`` and ``smlm-cluster``: the cluster centers have one
  more entry, ``Cluster centers``.

.. _migration-results:

Results that change
-------------------

With the same settings, these results differ from Picasso 0.11:

- **Defaults that were in camera pixels** are fixed lengths in nm, which no
  longer depend on the camera: linking 10 nm on the command line and in
  Python (Render used 10 nm already; ``picasso link`` used 1 camera pixel and
  ``postprocess.link`` 0.05 camera pixels), AIM 20 and 60 nm on the command
  line and in Python (were 20 and 60 nm only for a camera pixel size of
  130 nm), and the spacing of unfolded picks 250 nm (was 1 camera pixel).
- **RCC undrifting** renders the segments at 100 nm pixels and searches
  shifts up to 10 µm (0.11: at the camera pixel, up to 32 camera pixels), so
  the drift can differ slightly. ``picasso align`` and ``postprocess.align``
  repeat the alignment until the shifts converge.
- **NeNA** values change very slightly: 0.11 left out the last localizations.
- **Pick properties** hold the number of localizations of each pick in the
  ``locs`` column; 0.11 wrote the number of binding events there.
- **G5M** log-likelihoods are densities per nm² (per nm³ in 3D) instead of per
  camera pixel.
- **Simulate** simulates the photon rate it shows: with the default settings,
  0.11 simulated about 5,300 photons per frame while showing 22,750. Set
  *Photon detection rate* to about 300 for the previous brightness.

.. _migration-python:

In Python scripts
-----------------

Most scripts written for Picasso 0.11 need changes. Most calls that no longer
fit fail with an error (e.g., a ``TypeError`` for a removed argument or an
``AttributeError`` for a removed function), with two exceptions that give
wrong results instead: **numbers in camera pixels are taken as nm**, and a
**pixel size passed positionally** to a function that lost its ``pixelsize``
lands in the next argument (see :ref:`migration-python-units`). Check every
length your script passes. Also, the functions on localizations take only
``locs`` and ``info`` positionally and return the metadata of their result
(see :ref:`migration-python-locs-info`).

.. _migration-python-units:

Lengths in nm
~~~~~~~~~~~~~

``picasso.io.load_locs`` returns ``x``, ``y``, ``lpx``, ``lpy``, ``sx``,
``sy`` and every other length in nm, also for files written before 1.0, which
it converts. Every function takes and returns lengths in nm: pick positions
and sizes, radii, distances, bin sizes, drift, shifts, blur widths and
viewports. Counts of camera pixels that are not lengths stay as they are:
``Width`` and ``Height`` of the metadata, the box size, regions of interest
on the camera, and the sigma of the Gaussian filter in Localize.

To update a script, multiply the lengths it passes in camera pixels by the
pixel size, and remove any conversion of the columns to nm it does itself:

.. code-block:: python
   :caption: Picasso 0.11

   pixelsize = lib.get_from_metadata(info, "Pixelsize")
   x_nm = locs.x * pixelsize
   linked = postprocess.link(locs, info, r_max=20 / pixelsize)

.. code-block:: python
   :caption: Picasso 1.0

   x_nm = locs.x
   linked, linked_info = postprocess.link(locs, info, r_max=20)
   width_nm, height_nm = lib.field_of_view(info)  # Width, Height times Pixelsize

The arguments that took camera pixels, with their defaults:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Function and argument
     - Picasso 0.11
     - Picasso 1.0
   * - ``postprocess.link`` ``r_max``
     - 0.05 camera pixels
     - 10 nm
   * - ``postprocess.select_binding_event_cores`` ``r_max``
     - 0.1 camera pixels
     - 10 nm
   * - ``postprocess.picked_locs``, ``undrift_from_fiducials``
       ``pick_size`` of circles
     - radius, camera pixels
     - diameter, nm (see :ref:`below <migration-python-picks>`)
   * - ``postprocess.pair_correlation`` ``bin_size``, ``r_max``
     - camera pixels
     - nm
   * - ``postprocess.resi`` ``radius_xy``, ``radius_z``
     - camera pixels
     - nm
   * - ``aim.aim`` ``intersect_d``, ``roi_r``
     - 20/130 and 60/130 camera pixels
     - 20 and 60 nm
   * - ``clusterer.cluster`` ``radius_xy``, ``radius_z``;
       ``clusterer.dbscan`` ``radius``, ``radius_z``;
       ``clusterer.hdbscan`` ``cluster_eps``
     - camera pixels
     - nm
   * - ``g5m.g5m`` ``sigma_bounds`` with ``loc_prec_handle="abs"``
     - camera pixels
     - nm
   * - ``lib.unfold_localizations_square`` ``spacing``
     - 1 camera pixel
     - 250 nm
   * - ``render.render`` ``viewport``, ``min_blur_width``
     - camera pixels
     - nm (see :ref:`migration-python-render`)
   * - ``render.render_scene``, ``build_animation``: the viewports
     - camera pixels
     - nm
   * - ``nanotron.roi_to_img``, ``prepare_data``, ``predict_structure``
     - ``radius``, ``pick_radius`` (camera pixels)
     - ``pick_diameter`` (nm)

The results are in nm as well, e.g., the drift of ``postprocess.undrift``,
``undrift_from_fiducials`` and ``aim.aim``, the shifts of
``postprocess.align``, the result of ``postprocess.nena``, the pick centers
and box of ``imageprocess.find_fiducials``, and the statistics of
``postprocess.groupprops`` and ``clusterer.find_cluster_centers`` (``area``
in nm², ``volume`` in nm³).

**Removed** ``pixelsize`` **arguments.** Functions that took the pixel size
only to convert between camera pixels and nm lost it:
``clusterer.cluster``, ``dbscan``, ``hdbscan``, ``find_cluster_centers``,
``postprocess.plot_drift``, ``aim.intersection_max_z``,
``render.render_hist3d``, ``render_hist3d_anisotropic``,
``render.draw_scalebar``, ``draw_points`` and ``spinna.coords_to_locs``.
Remove it from your calls. **Watch out for positional calls:** the pixel size
then lands in the next argument without an error, e.g.,
``render.draw_scalebar(image, viewport, 500, 130)`` passes 130 as
``display_length``. The clustering functions take their parameters by name
only (see :ref:`migration-python-locs-info`), so positional calls of them fail
with a ``TypeError``.

.. code-block:: python
   :caption: Picasso 0.11

   linked = postprocess.link(locs, info, r_max=0.05)
   clustered = clusterer.cluster(
       locs, 0.15, 10, True, radius_z=0.4, pixelsize=130, return_info=False
   )
   centers = clusterer.find_cluster_centers(clustered, pixelsize=130)

.. code-block:: python
   :caption: Picasso 1.0

   linked, linked_info = postprocess.link(locs, info, r_max=6.5)
   clustered, clustered_info = clusterer.cluster(
       locs, info, radius_xy=20, min_locs=10, frame_analysis=True, radius_z=50
   )
   centers, centers_info = clusterer.find_cluster_centers(
       clustered, clustered_info
   )

.. _migration-python-picks:

Circular picks by their diameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every function takes the size of circular picks as their **diameter**, as
Render and the pick region files do. ``picasso.postprocess.picked_locs`` and
``undrift_from_fiducials`` took the radius in camera pixels:

.. code-block:: python
   :caption: Picasso 0.11

   picked = postprocess.picked_locs(locs, info, picks, "Circle", pick_size=0.77)

.. code-block:: python
   :caption: Picasso 1.0

   picks, shape, size = io.load_picks("locs_picks.yaml")  # positions and size in nm
   picked = postprocess.picked_locs(
       locs, info, picks=picks, pick_shape="Circle", pick_size=200
   )

The functions of ``picasso.nanotron`` take ``pick_diameter`` (nm) instead of
``radius`` and ``pick_radius``. ``picasso.io.load_picks`` returns the
positions and the size in nm; its ``pixelsize`` argument is only needed for
pick files written before 1.0, which it converts (in 0.11, it converted the
size to camera pixels).

.. _migration-python-locs-info:

Localizations, metadata and the other arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The functions on localizations share one pattern (see :ref:`api-locs-info`):

- **Only the localizations and their metadata are positional**,
  ``f(locs, info, *, ...)`` (``f(locs, infos, *, ...)`` for several channels;
  writers such as ``io.save_locs`` take the path first). Pass every other
  argument by name; positional calls fail with a ``TypeError``, e.g.,
  ``postprocess.link(locs, info, 10, 3)``, ``postprocess.undrift(locs, info,
  1000)`` or ``zfit.calibrate_z(locs, info, 10, 0.79)``. Functions that do
  not need the metadata take ``info=None``, so it can be left out, e.g.,
  ``clusterer.dbscan(locs, radius=20, min_samples=4)``.
- **Functions that return new localizations return their metadata, too**: the
  complete metadata of the result, i.e., ``info`` with an entry recording the
  step appended, ready to be saved. ``info`` itself is not modified. Further
  results come after the localizations and the metadata. To record a step of
  your own, use ``lib.append_metadata``.

The functions whose results changed:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Function
     - Picasso 0.11 returned
     - Picasso 1.0 returns
   * - ``postprocess.link``, ``select_binding_event_cores``,
       ``combine_locs_in_picks``, ``remove_locs_in_picks``,
       ``compute_dark_times``, ``compute_local_density``, ``apply_drift``
     - ``locs``
     - ``locs, info``
   * - ``postprocess.undrift``
     - ``drift, locs``
     - ``locs, info, drift``
   * - ``postprocess.align``, ``align_rcc``, ``align_from_picked``
     - ``locs`` or ``locs, shifts``
     - ``locs, infos`` or ``locs, infos, shifts``
   * - ``clusterer.cluster``, ``dbscan``, ``hdbscan``
     - ``locs, entry`` (one dictionary)
     - ``locs, info`` (the whole list)
   * - ``clusterer.find_cluster_centers``
     - ``centers``
     - ``centers, info``
   * - ``g5m.g5m``
     - ``centers, clustered_locs, info``
     - ``centers, info, clustered_locs``
   * - ``average.average``
     - ``locs``, or ``locs, info`` with ``return_shifted_locs=True``
     - ``locs, info``
   * - ``lib.apply_filter_steps``
     - ``locs, ranges, to_remove, missing``
     - ``locs, info, missing``

``aim.aim``, ``postprocess.undrift_from_fiducials``, ``zfit.zfit``,
``lib.unfold_localizations_square`` and the other functions that returned
the metadata in 0.11 return it as before, ``unfold_localizations_square``
with an entry recording the unfolding.

``lib.apply_filter_steps`` takes the metadata of the localizations as
``info`` and the metadata with the filter steps as ``recipe``; in 0.11, the
filter steps came second. ``postprocess.remove_locs_in_picks`` and
``compute_dark_times`` no longer modify the localizations passed to them.

.. code-block:: python
   :caption: Picasso 0.11

   drift, undrifted = postprocess.undrift(locs, info, 1000)
   linked = postprocess.link(undrifted, info, 0.05, 3)
   info.append({"Generated by": "Picasso Link", "Maximum Distance": 0.05})
   clustered, dbscan_info = clusterer.dbscan(linked, 0.15, 4)
   io.save_locs("locs_dbscan.hdf5", clustered, info + [dbscan_info])
   filtered, ranges, to_remove, missing = lib.apply_filter_steps(
       locs, filter_info
   )

.. code-block:: python
   :caption: Picasso 1.0

   undrifted, info, drift = postprocess.undrift(
       locs, info, segmentation=1000
   )
   linked, info = postprocess.link(undrifted, info, r_max=6.5, max_dark_time=3)
   clustered, info = clusterer.dbscan(linked, info, radius=20, min_samples=4)
   io.save_locs("locs_dbscan.hdf5", clustered, info)  # with all three steps
   filtered, filtered_info, missing = lib.apply_filter_steps(
       locs, info, recipe=filter_info
   )

.. _migration-python-localize:

Localizing
~~~~~~~~~~

**The identification settings** are one
:class:`picasso.localize.IdentificationParameters`: the ``method``
(:class:`picasso.localize.NetGradient`, the new
:class:`picasso.wavelet.WaveletParameters`, or a list of them, one per region
of interest), the ``box`` and the filters (``temporal_median_window``,
``temporal_median_stride``, ``gaussian_filter_sigma``).
:meth:`~picasso.localize.IdentificationParameters.from_info` reads them from
the metadata of a file, and
:meth:`~picasso.localize.IdentificationParameters.to_info` writes them; the
metadata keys are unchanged.

**The camera information** needs ``Pixelsize`` (nm): ``localize`` and ``fit``
raise a ``ValueError`` without it (0.11 assumed 130 nm). ``Qe`` is ignored,
and the ``camera_parameters`` of the movie classes in ``picasso.io`` no longer
return ``qe``.

.. code-block:: python
   :caption: Picasso 0.11

   camera_info = {"Baseline": 100, "Sensitivity": 0.5, "Gain": 1}
   ids, identify_info = localize.identify(movie, 5000, 7)
   locs, info = localize.localize(
       movie,
       camera_info=camera_info,
       identification_parameters={"Min. Net Gradient": 5000, "Box Size": 7},
       movie_info=movie_info,
       fit_progress_callback="console",
   )

.. code-block:: python
   :caption: Picasso 1.0

   camera_info = {"Baseline": 100, "Sensitivity": 0.5, "Gain": 1, "Pixelsize": 130}
   parameters = localize.IdentificationParameters(
       method=localize.NetGradient(5000), box=7
   )
   ids, identify_info = localize.identify(movie, parameters=parameters)
   locs, info = localize.localize(
       movie,
       camera_info=camera_info,
       identification_parameters=parameters,
       movie_info=movie_info,
       progress="console",
   )

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Picasso 0.11
     - Picasso 1.0
   * - ``localize.identify(movie, minimum_ng, box, temporal_median_window=...,
       gaussian_filter_sigma=..., return_info=...)``
     - ``localize.identify(movie, parameters=IdentificationParameters(...))``,
       which always returns ``(identifications, info)``.
   * - ``localize.localize`` with a dictionary as ``identification_parameters``
       (or ``parameters``), ``mle_method`` and ``return_info``
     - ``localize.localize(movie, camera_info=...,
       identification_parameters=IdentificationParameters(...))``, with only
       the movie positionally, which always returns ``(locs, info)``.
   * - ``localize.fit2D(movie, movie_info, camera_info, identifications,
       box)``
     - ``localize.fit(movie, camera_info=..., identifications=...,
       box=...)``
   * - ``localize.localize_3D(movie, ..., box=7, minimum_ng=5000,
       calibration_3d=..., multiprocess=...)``
     - ``localize.localize(movie, ..., identification_parameters=...,
       calibration_3d=..., threaded=...)``
   * - ``localize.identify_multichannel_sum(movies, minimum_ng, box,
       transforms, temporal_median_window=..., ...)``
     - ``localize.identify_multichannel_sum(movies, transforms,
       parameters=IdentificationParameters(...))``
   * - ``localize.identify_in_frame``, ``identify_by_frame_number`` and
       ``identify_async`` with ``minimum_ng`` (a number, or a list per region)
     - The method in its place: ``NetGradient(5000)``, or a list of them.
   * - ``localize.locs_from_fits``, ``locs_from_fits_gauss_gpu``
     - ``localize.locs_from_fits_gauss``
   * - ``localize.locs_from_fits_gauss(identifications, theta, box, em)``
       (camera pixels)
     - The same with ``pixelsize=`` (keyword-only, required), returning nm;
       so do ``locs_from_fits_spline`` and
       ``locs_from_fits_gauss_multichannel``.
   * - ``localize.picks_to_identifications(picks)``,
       ``locs_to_identifications``, ``add_roi_id``,
       ``locs_to_region_coordinates`` and ``split_locs_by_region`` (camera
       pixels)
     - The same with the pixel size, to place the nm positions on the camera:
       ``pixelsize=`` (``locs_to_region_coordinates(locs, region,
       pixelsize)``).
   * - ``localize.check_nena(locs, info, callback=...)``, ``check_drift``
       (camera pixels)
     - ``check_nena(locs, info, progress=...)``, ``check_drift``, in nm.

**The modules** ``picasso.gausslq`` **and** ``picasso.gaussmle`` are removed;
all fitting is in ``picasso.localize`` and ``picasso.fitting``. The fitting
methods ``"gausslq"`` and ``"gaussmle"`` of ``localize.fit`` and
``localize.localize`` are unchanged.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Picasso 0.11
     - Picasso 1.0
   * - ``gausslq.fit_spots``, ``fit_spots_parallel``, ``fit_spot``
     - ``localize.fit_spots_gauss(spots)``
   * - ``gausslq.fit_spots_gauss_gpu``
     - ``localize.fit_spots_gauss(spots, use_gpu=True)``
   * - ``gaussmle.gaussmle(spots, eps, max_it)``
     - ``localize.fit_spots_gauss(spots, mle=True, return_stats=True)``,
       which returns ``(theta, log_likelihood, iterations, chi_square)``; the
       precisions are computed by ``locs_from_fits_gauss(..., mle=True)``.
   * - ``gausslq.locs_from_fits``, ``gausslq.locs_from_fits_gauss_gpu``,
       ``gaussmle.locs_from_fits``
     - ``localize.locs_from_fits_gauss(..., pixelsize=...)``, with
       ``mle=True``, ``log_likelihood`` and ``iterations`` for MLE fits.
   * - ``gausslq.fit_spots_parallel(spots, asynch=True)``,
       ``gausslq.fits_from_futures``, ``gaussmle.gaussmle_async``
     - ``fitting.gaussfit.fit_spots_async(model, spots, initial_parameters)``
       and its ``results()``.
   * - ``gausslq.localization_precision``, ``gausslq.sigma_uncertainty``,
       ``gaussmle.sigma_uncertainty``
     - ``fitting.precision.localization_precision``,
       ``sigma_uncertainty_lsq``, ``sigma_uncertainty_mle``. They take the
       spot widths in camera pixels: divide the nm columns by the pixel size
       and multiply the result by it.

The parameters returned by the fits (``theta``) are ordered
``photons, x, y, sx, sy, bg`` (0.11: ``x, y, photons, bg, sx, sy``).

.. code-block:: python
   :caption: Picasso 0.11

   from picasso import gausslq, gaussmle

   theta = gausslq.fit_spots_parallel(spots, asynch=False)
   locs = gausslq.locs_from_fits(ids, theta, 7, em=False)
   theta, crlbs, log_l, iters = gaussmle.gaussmle(spots, eps=0.001, max_it=1000)
   locs = gaussmle.locs_from_fits(ids, theta, crlbs, log_l, iters, 7)

.. code-block:: python
   :caption: Picasso 1.0

   from picasso import localize

   theta = localize.fit_spots_gauss(spots)
   locs = localize.locs_from_fits_gauss(ids, theta, 7, em=False, pixelsize=130)
   theta, log_l, iters, _ = localize.fit_spots_gauss(spots, mle=True, return_stats=True)
   locs = localize.locs_from_fits_gauss(
       ids, theta, 7, em=False, mle=True,
       log_likelihood=log_l, iterations=iters, pixelsize=130,
   )

.. _migration-python-calibrations:

Calibrations
~~~~~~~~~~~~

The calibrations take the identification method as ``identification_method``
(e.g., ``NetGradient(5000)``, or one per channel) in place of ``minimum_ng``:
``picasso.spline.calibrate_spline``, ``calibrate_spline_multichannel``,
``calibrate_spline_split_fov``, ``refine_split_fov_transforms_from_signal``,
``refine_multichannel_transforms_from_signal``,
``picasso.registration.calibrate_channel_registration_from_beads``,
``calibrate_channel_registration_from_signal``,
``picasso.localize.fit_lateral_transform`` and ``calibrate_lateral_transform``.
The spline calibrations need ``Pixelsize`` in ``camera_info``, and the lateral
calibrations take a required ``pixelsize``; the lateral corrections they
store are in nm.

.. code-block:: python
   :caption: Picasso 0.11

   calibration = spline.calibrate_spline(
       movie, info, camera_info, box=13, minimum_ng=5000, d=20
   )
   calibration, qc = localize.fit_lateral_transform(
       movie_ref, movie_target, {}, box=7, minimum_ng=5000,
       transform_type="chromatic",
   )

.. code-block:: python
   :caption: Picasso 1.0

   calibration = spline.calibrate_spline(
       movie, info, camera_info, box=13,
       identification_method=localize.NetGradient(5000), d=20,
   )
   calibration, qc = localize.fit_lateral_transform(
       movie_ref, movie_target, {}, box=7,
       identification_method=localize.NetGradient(5000), pixelsize=130,
       transform_type="chromatic",
   )

The astigmatism (3D) calibrations of ``picasso.zfit.calibrate_z`` give the
spot widths in nm (0.11: camera pixels) and record ``picasso_format``.
``zfit.zfit`` and ``localize.localize`` convert a calibration without
``picasso_format`` (written before 1.0) with the pixel size of the
localizations, so a calibration you build by hand in nm needs
``"picasso_format": "1.0"``. ``picasso.io.load_calibration`` takes
``pixelsize`` for calibration files written before 1.0.

``picasso.transforms.Transform.decompose`` takes ``unit="nm"`` (the default)
or ``"px"`` instead of ``pixelsize`` and returns ``tx_nm``/``ty_nm`` or
``tx_px``/``ty_px``; its first argument is ``at``. The channel registrations
stay in camera pixels, so decompose them with ``unit="px"``.

.. _migration-python-progress:

Progress
~~~~~~~~

Every function that reports its progress takes one ``progress`` argument:
``"console"`` for a progress bar per step, None (the default) for nothing, or
a progress tracker such as ``picasso.lib.ProgressDialog`` (any object with the
:class:`picasso.lib.Progress` interface, e.g., a subclass of
:class:`picasso.lib.MockProgress`). The function names and sizes its steps
itself. This replaces ``progress_callback``, ``callback`` and
``callback_parent``, and the separate callbacks per step of functions with
several steps:

.. code-block:: python
   :caption: Picasso 0.11

   locs, info = localize.localize(
       movie, camera_info=camera_info, identification_parameters=parameters,
       identification_progress_callback="console",
       fit_progress_callback="console",
   )
   picked = postprocess.picked_locs(locs, info, picks, "Circle", 0.77, callback="console")

.. code-block:: python
   :caption: Picasso 1.0

   locs, info = localize.localize(
       movie, camera_info=camera_info, identification_parameters=parameters,
       progress="console",
   )
   picked = postprocess.picked_locs(
       locs, info, picks=picks, pick_shape="Circle", pick_size=200,
       progress="console",
   )

- ``picasso.io.load_locs``, ``load_movie``, ``load_movie_all``, ``load_tif``
  and ``load_tif_concatenated`` take a tracker instead of a function called
  with ``(done, total)``.
- ``picasso.g5m.g5m`` takes ``progress`` instead of ``callback_parent`` and is
  silent by default (0.11: a console progress bar).
- ``picasso.average.average`` reports its progress through ``progress`` and
  passes the localizations after each iteration to the new
  ``iteration_callback(iterations_done, locs)``, instead of one function with
  five arguments.
- The methods of ``picasso.spinna.SPINNA``, ``spinna.compare_models`` and
  ``fit_le`` take ``progress`` instead of ``callback``, with the same values.

.. _migration-python-postprocess:

Postprocessing
~~~~~~~~~~~~~~

**Alignment.** ``postprocess.align`` is the main alignment function, with
``method="rcc"`` (the default) or ``method="picked"`` (with ``picks``,
``pick_shape`` and ``pick_size``, as ``align_from_picked``). Compared with
0.11:

- ``align`` repeats the RCC alignment until the shifts converge, as Render
  does (0.11: a single pass).
- ``align`` and ``align_from_picked`` no longer shift the input DataFrames in
  place; use the returned ones. The ``apply_shifts`` argument of ``align`` is
  removed, and its arguments after ``infos`` are keyword-only.
- The shifts returned with ``return_shifts=True`` are ordered x first,
  ``(shift_x, shift_y[, shift_z])``, in nm. ``align_from_picked`` returned
  them y first, and ``align_rcc`` returned the mean shift per iteration
  instead of the shift of each channel.

.. code-block:: python
   :caption: Picasso 0.11

   aligned, (shift_y, shift_x) = postprocess.align_from_picked(
       [locs1, locs2], [info1, info2],
       picks=picks, pick_shape="Circle", pick_size=1.54, return_shifts=True,
   )

.. code-block:: python
   :caption: Picasso 1.0

   aligned, aligned_infos, (shift_x, shift_y) = postprocess.align(
       [locs1, locs2], [info1, info2], method="picked",
       picks=picks, pick_shape="Circle", pick_size=200, return_shifts=True,
   )

**RCC undrifting.** ``postprocess.undrift`` takes one ``progress`` instead of
``segmentation_callback`` and ``rcc_callback``, and returns the drift in nm.
It renders the segments at 100 nm pixels and searches shifts up to 10 µm
(``disp_px_size`` and ``max_shift``, keyword-only); 0.11 rendered at the
camera pixel and searched up to 32 camera pixels. To repeat a 0.11 result
exactly:

.. code-block:: python

   pixelsize = lib.get_from_metadata(info, "Pixelsize")
   locs, info, drift = postprocess.undrift(
       locs, info, segmentation=1000, display=False,
       disp_px_size=pixelsize, max_shift=32 * pixelsize,
   )

**Pick properties.** ``postprocess.pick_properties`` takes one ``progress``
instead of ``kinetics_progress`` and ``groupprops_progress``. Its ``locs``
column holds the number of localizations in each pick; 0.11 wrote the number
of binding events there (as in ``n_events``).

**Removed:** ``postprocess.cluster_combine`` and ``cluster_combine_dist``.

.. _migration-python-render:

Rendering
~~~~~~~~~

``picasso.render`` is a package; every function is still imported from
``picasso.render``.

``render.render`` takes ``viewport`` (``((y_min, x_min), (y_max, x_max))``)
and ``min_blur_width`` in nm, as ``disp_px_size`` already was:

.. code-block:: python
   :caption: Picasso 0.11

   n, image = render.render(
       locs, info, disp_px_size=10, viewport=((10, 10), (20, 20)),
       blur_method="gaussian", min_blur_width=0.1,
   )

.. code-block:: python
   :caption: Picasso 1.0

   n, image = render.render(
       locs, info, disp_px_size=10, viewport=((1300, 1300), (2600, 2600)),
       blur_method="gaussian", min_blur_width=13,
   )

- **Relative intensities.** ``render.render_scene`` and ``build_animation``
  take ``channel_contrast``, one ``(low, high)`` window per channel relative
  to the global contrast limits, instead of ``relative_intensities``. A
  relative intensity ``r`` of 1 or more corresponds to ``(0, 1 / r)``;
  intensities below 1, which capped the brightest color of a channel, have no
  equivalent. ``render.scale_intensities`` is removed.
- **Animations.** ``render.build_animation`` no longer accepts positions as
  Euler angles ``(angle_x, angle_y, angle_z, viewport)``; pass
  ``(rotation, viewport)`` with a ``scipy.spatial.transform.Rotation``, which
  ``render.rotation_matrix(angle_x, angle_y, angle_z)`` makes from the old
  angles. It takes ``progress`` instead of ``progress_callback`` and returns
  whether the animation was completed.
- **Histograms.** ``render.render_hist3d`` and ``render_hist3d_anisotropic``
  lost ``pixelsize``, and, with ``render.render_hist`` (was
  ``render_hist_numba``), take the coordinates and bounds in nm and
  ``oversampling`` in pixels per nm, i.e., 1 / (pixel size in nm).
- **Overlays.** ``render.draw_scalebar`` and ``draw_points`` lost
  ``pixelsize`` and take nm; ``render.optimal_scalebar_length`` takes the
  width of the image in nm instead of ``(pixelsize, width)``.
- **Imaris export.** ``ext.bitplane.numpy_to_imaris`` takes the rendered
  pixel size ``disp_px_size`` (nm) instead of ``oversampling`` and
  ``pixelsize``.

.. code-block:: python
   :caption: Picasso 0.11

   qimage, n = render.render_scene(
       [locs1, locs2], [info1, info2], disp_px_size=10,
       relative_intensities=[1, 2],
   )
   positions = [(0, 0, 0, viewport), (0, np.pi / 2, 0, viewport)]

.. code-block:: python
   :caption: Picasso 1.0

   qimage, n = render.render_scene(
       [locs1, locs2], [info1, info2], disp_px_size=10,
       channel_contrast=[(0, 1), (0, 1 / 2)],
   )
   positions = [
       (render.rotation_matrix(0, 0, 0), viewport),
       (render.rotation_matrix(0, np.pi / 2, 0), viewport),
   ]

.. _migration-python-io:

Loading and saving
~~~~~~~~~~~~~~~~~~

- ``picasso.io.load_locs``, ``load_info``, ``load_filter``,
  ``load_identifications`` and ``load_mask`` no longer take ``qt_parent``:
  they open no dialogs and print nothing; their errors say what is wrong.
  The arguments of ``load_locs`` after the path are keyword-only.
- ``load_locs`` raises ``picasso.io.LocsFormatError`` (a ``ValueError``)
  instead of a ``KeyError`` when a file lacks a required metadata key or
  column, and ``picasso.io.MissingPixelsizeError`` (a ``LocsFormatError``)
  when only the pixel size is missing; pass it as ``pixelsize=``.
- ``picasso.io.load_clusters`` is removed: cluster centers are localization
  files, so use ``picasso.io.load_locs``. The cluster tables written before
  Picasso 0.5 are no longer read.
- Metadata read from ``.yaml`` files holds plain Python values (``float``,
  ``int``, lists) instead of NumPy scalars and arrays, as the metadata stored
  in ``.hdf5`` files already did.
- ``picasso.spinna.load_structures`` is ``picasso.io.load_structures``.
- ``picasso.spinna.coords_to_locs`` lost its ``pixelsize``, and
  ``StructureSimulator.save`` and ``StructureMixer.save`` need the pixel size
  (``pixelsize``, or a mask) instead of assuming 130 nm.

.. _migration-python-renamed:

Renamed functions
~~~~~~~~~~~~~~~~~

The old names are removed, without aliases.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Picasso 0.11
     - Picasso 1.0
   * - ``picasso.simulate``: ``defineStructure``, ``generatePositions``,
       ``prepareStructures``, ``distphotons``, ``distphotonsxy``,
       ``rotateStructure``, ``incorporateStructure``, ``randomExchange``,
       ``convertMovie``, ``saveMovie``, ``saveInfo``
     - ``define_structure``, ``generate_positions``, ``prepare_structures``,
       ``distribute_photons``, ``distribute_photons_xy``,
       ``rotate_structure``, ``incorporate_structure``, ``random_exchange``,
       ``convert_movie``, ``save_movie``, ``save_info``.
       ``distribute_photons_xy`` and ``convert_movie`` take ``pixelsize``
       for 3D (astigmatism) simulations.
   * - ``picasso.design``: ``convertPlateIndex``, ``convertPlateIndexColor``,
       ``readPlate``, ``savePlate``, ``saveInfo``
     - ``convert_plate_index``, ``convert_plate_index_color``,
       ``read_plate``, ``save_plate``, ``save_info``
   * - ``picasso.lib``: ``hist2d_numba``, ``is_loc_in_square_numba``,
       ``locs_in_square_numba``, ``is_loc_in_box_numba``,
       ``locs_in_box_numba``
     - ``hist2d``, ``is_loc_in_square``, ``locs_in_square``,
       ``is_loc_in_box``, ``locs_in_box``
   * - ``picasso.lib``: ``is_loc_at_numba``, ``locs_at_numba``,
       ``is_loc_in_rectangle_numba``, ``locs_in_rectangle_numba``
     - ``is_loc_in_circle``, ``locs_in_circle``,
       ``is_loc_in_oriented_rectangle``, ``locs_in_oriented_rectangle``
       (``lib.is_loc_at``, ``locs_at`` and ``locs_in_rectangle`` are
       unchanged)
   * - ``picasso.postprocess.get_block_locs_at_numba``
     - ``picasso.postprocess.get_block_locs_at``
   * - ``picasso.render.render_hist_numba``
     - ``picasso.render.render_hist``
   * - ``picasso.spinna.load_structures``
     - ``picasso.io.load_structures``
   * - ``picasso.lib``: ``get_sound_notification_path``,
       ``get_available_sound_notifications``, ``set_sound_notification``,
       ``SOUND_NOTIFICATION_DURATION``
     - ``picasso.notifications``: ``selected_sound`` (the file name),
       ``available_sounds``, ``select_sound``, ``MIN_DURATION``

.. _migration-python-removed:

Removed functions
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Picasso 0.11
     - Picasso 1.0
   * - ``picasso.gausslq``, ``picasso.gaussmle``, ``localize.fit2D``,
       ``localize.localize_3D``, ``localize.locs_from_fits``,
       ``localize.locs_from_fits_gauss_gpu``
     - See :ref:`migration-python-localize`.
   * - The ``return_info`` argument of ``localize.identify``,
       ``localize.localize``, ``clusterer.cluster``, ``dbscan`` and
       ``hdbscan``
     - Removed; they always return ``(locs, info)``.
   * - ``postprocess.cluster_combine``, ``cluster_combine_dist``
     - Removed.
   * - ``io.load_clusters(path)``
     - ``io.load_locs(path)``
   * - ``lib.append_to_rec(locs, data, name)``
     - ``locs[name] = data``
   * - ``lib.remove_from_rec(locs, name)``
     - ``locs = locs.drop(columns=name)``
   * - ``render.scale_intensities``
     - ``channel_contrast`` of ``render.render_scene``, see
       :ref:`migration-python-render`.
   * - ``lib.ProgressType``
     - ``lib.Progress``
   * - ``lib.open_sound_notifications_folder``
     - ``notifications.sounds_directory()`` returns the folder.

.. _migration-python-messages:

Warnings and messages
~~~~~~~~~~~~~~~~~~~~~

- Deprecation warnings are ``FutureWarning`` instead of
  ``DeprecationWarning``, so that scripts show them. Code that filters
  ``DeprecationWarning`` to silence Picasso has to filter ``FutureWarning``
  instead.
- Picasso reports through Python's :mod:`logging` instead of printing (see
  :doc:`api/index`): ``logging.getLogger("picasso")`` shows, silences or
  redirects its messages. Output asked for with ``verbose=True`` is still
  printed.

.. _migration-plugins:

In plugins
----------

The three hooks of a plugin (the ``Plugin`` class, ``PICASSO_API`` and
``register_cli``) are unchanged, see :ref:`plugins-for-developers`. A plugin
that calls the Python API needs the changes above. GUI plugins that use the
attributes of the Render window need these:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Picasso 0.11
     - Picasso 1.0
   * - ``View.pixelsize``, ``DisplaySettingsDialog.pixelsize``
     - ``View.pixelsizes``, the read-only pixel size of each channel (from
       its metadata), and ``View.default_pixelsize()``
   * - ``DisplaySettingsDialog.zoom``, ``View.set_zoom``
     - ``DisplaySettingsDialog.display_width`` (µm)
   * - ``View.viewport``, the localizations (camera pixels)
     - The same in nm
   * - ``DatasetDialog.intensitysettings``
     - ``DatasetDialog.contrast_low`` and ``contrast_high`` (% of the global
       contrast)
   * - ``View.read_relative_intensities()``
     - ``View.read_channel_contrast()``, a ``(low, high)`` window per channel
   * - ``Window.fast_render_dialog``, ``View.fast_render_indices``
     - Removed
   * - ``io.load_locs(path, qt_parent=self)``, which showed errors in a
       message box
     - ``io.load_locs(path)``; catch its errors and show them yourself.
   * - ``progress_callback`` or ``callback`` functions that update a
       ``lib.ProgressDialog``
     - The ``lib.ProgressDialog`` itself as ``progress``, see
       :ref:`migration-python-progress`

The other attributes of the windows are not public and may change in minor
releases; check GUI plugins against each release (see :doc:`versioning`).

.. _migration-installing:

Installing
----------

- Picasso 1.0 needs Python 3.11 or later (0.11: 3.10).
- Picasso Server is an optional extra of the Python package, as it alone needs
  Streamlit and Plotly: ``pip install "picassosr[server]"``. The one-click
  installers include it.
- The Windows installer's *GPU* edition is called *CUDA*.
