File Formats
============

This section describes the different file format and name conventions used in
Picasso.

.. _files-movie:

Movie Files
-----------

Picasso accepts several types of raw movie files:

- TIFF-family stacks (``.tif``/``.tiff``/``.ome.tif``/BigTIFF/Zeiss ``.lsm``);
- raw binary data (file extension ``.raw``);
- the Nikon format ``.nd2``;
- MetaMorph ``.stk``;
- Imaris ``.ims``;
- with the optional ``czi``/``lif`` extras (Python ≥ 3.12), Zeiss ``.czi``
  and Leica ``.lif``.

For TIFF-family, ``.nd2`` and ``.stk`` movies the metadata is normally read
directly from the file. Raw binary files carry no metadata, so
``Picasso: Localize`` asks for it (``Frames``, ``Width``, ``Height``,
``Pixelsize``, etc.) if sidecar ``.yaml`` file is not found; it does the same for other movies whose metadata cannot
be parsed, with the field of view dimensions pre-filled when they can be read.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``io.load_movie`` opens any supported movie; ``io.save_raw`` writes a
   ``.raw`` movie with its ``.yaml``.

   .. code-block:: python

      from picasso import io

      movie, info = io.load_movie("movie.tif")
      io.save_raw("movie.raw", movie[:1000], info[0])


.. _files-hdf5:

HDF5 Files
----------

HDF5 is a generic and efficient binary file format for storing data. An HDF5
file can hold several datasets, each accessible by a path within the file,
similar to a path of an operating system. Picasso saves the following
``.hdf5`` files:

- **Localizations**, including molecular maps, as a table under ``/locs``; see
  :ref:`files-localization-hdf5` and :ref:`files-molecular-maps`.
- **Pick properties**, as a table under ``/groups``; see
  :ref:`files-pick-properties`.
- **Spot identifications** from Localize, as a table under
  ``/identifications``; see :ref:`localize-saving-loading-identifications`.
- **Cubic-spline PSF calibrations** (``*_spline_calib.hdf5``), see
  :ref:`localize-spline-building`, and **sCMOS camera calibrations**
  (``*_scmos_calib.hdf5``), see :ref:`localize-scmos-calibration`. These
  store arrays (the spline coefficients or the per-pixel maps) as datasets
  and the remaining parameters as a JSON string in the file attribute
  ``metadata``; they are read with ``picasso.io.load_spline_calibration`` and
  ``picasso.io.load_camera_calibration``.

The localization files also embed their metadata under the path
``/metadata`` (see :ref:`files-metadata-settings` below), so that the file is
self-contained. Files of 100,000 localizations or more also store Render's
spatial index under ``/render_index`` (see :ref:`spatial-index` below).

An HDF5 file can be opened with various software packages. In Picasso, we use
``pandas`` for this purpose. For example, to open localizations:

.. code-block:: python

   import pandas

   locs = pandas.read_hdf(PATH_TO_LOCALIZATIONS, key="locs")

The ``key`` argument can be adjusted for other datasets. The available keys
can be verified using ``pandas.HDFStore(PATH_TO_FILE).keys()``. To write
localizations, use ``picasso.io.save_locs`` rather than pandas, see
:ref:`files-minimum-requirements`.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``io.load_locs`` and ``io.save_locs`` read and write the table together
   with its metadata. Other datasets (``groups``, ``identifications``, ...)
   are read with ``pandas``.

   .. code-block:: python

      import pandas as pd
      from picasso import io

      locs, info = io.load_locs("movie_locs.hdf5")
      io.save_locs("movie_locs_copy.hdf5", locs, info)

      with pd.HDFStore("movie_locs_pickprops.hdf5") as store:
          print(store.keys())
      groups = pd.read_hdf("movie_locs_pickprops.hdf5", key="groups")


.. _files-minimum-requirements:

Minimum requirements
~~~~~~~~~~~~~~~~~~~~

.. important::

   To open localizations from other software in Picasso
   (``locs, info = picasso.io.load_locs(path)``, Render, Filter, etc.), the HDF5
   file needs:

   1. **A table under** ``/locs`` with at least the columns ``frame`` (starting
      at 0), ``x``, ``y``, ``lpx`` and ``lpy`` (all in nm; see
      :ref:`Table 1 <files-localization-columns>`). 3D data also needs ``z`` (in
      nm); ``lpz`` (in nm) is optional but recommended for 3D rendering.
   2. **Metadata**, embedded under ``/metadata`` or in a YAML file of the same
      name (see :ref:`files-metadata-settings`), with the
      keys:

      - ``Width`` and ``Height``: size of the camera image in camera pixels;
      - ``Frames``: number of frames in the movie;
      - ``Pixelsize``: camera pixel size in nm. The field of view spans
        ``Width × Pixelsize`` by ``Height × Pixelsize`` nm.
   3. **The format marker**: the HDF5 root attribute ``picasso_format`` set to
      ``"1.0"`` (see :ref:`files-format-versions`). A file without it counts
      as written before Picasso 1.0, with lengths in camera pixels, and is
      converted to nm when it is opened (see :ref:`files-legacy`).

      Picasso searches the metadata list from the last entry to the first, so
      the keys can be in any entry; usually, they are in the first one (the
      first document of the YAML file), which describes the movie. A minimal
      YAML file, with a second document recording a later processing step,
      looks like this:

      .. code-block:: yaml

         Width: 256
         Height: 256
         Frames: 10000
         Pixelsize: 130
         ---
         Generated by: my_analysis_script.py
         Drift correction: true

Loading fails with a ``picasso.io.LocsFormatError`` that names what is missing
if the file lacks one of the columns ``frame``, ``x``, ``y``, ``lpx`` and
``lpy`` or one of the metadata keys ``Width``, ``Height``, ``Frames`` and
``Pixelsize``, and if frame numbers are negative (frames count from 0;
fractional frames, such as the mean frames of cluster centers, are allowed).
If only ``Pixelsize`` is missing, the error is a
``picasso.io.MissingPixelsizeError`` (a ``LocsFormatError``), and the GUIs ask
for the pixel size instead of failing. Without metadata at all, a
``NoMetadataFileError`` is raised.

Localizations with NaN or infinite values in any column, negative ``x``, ``y``
or precisions, or ``x`` ≥ ``Width × Pixelsize`` or ``y`` ≥
``Height × Pixelsize`` are discarded on loading. Picasso reports how many were
discarded (Render shows a message): if most of them lie beyond the field of
view, ``Width``, ``Height`` or ``Pixelsize`` are too small, or the file holds
camera pixels although it carries the format marker.

.. important::

   **From Python, always save localizations with**
   ``picasso.io.save_locs(path, locs, info)``, with ``locs`` a
   ``pandas.DataFrame`` and ``info`` a list of dictionaries, e.g.,
   ``[{"Width": 256, "Height": 256, "Frames": 10000, "Pixelsize": 130}]``.
   It writes everything listed above in one step: the table in the layout
   every Picasso version reads, the format marker, the embedded metadata
   (and the YAML file, if one is there or the settings ask for it), and the
   :ref:`spatial index <spatial-index>`. It also casts the columns to the
   :ref:`expected data types <files-dtypes>` and drops localizations that
   Picasso would discard on loading.

   Do not write the file with ``pandas.DataFrame.to_hdf`` or ``h5py``
   yourself. Such a file has no format marker, so Picasso takes its nm for
   camera pixels and multiplies them by the pixel size, and its metadata
   is easily left out or left stale.

From MATLAB, see :ref:`files-matlab-export`.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      import pandas as pd
      from picasso import io

      # x, y, lpx and lpy in nm
      locs = pd.DataFrame({"frame": frame, "x": x, "y": y, "lpx": lpx, "lpy": lpy})
      info = [
          {"Width": 256, "Height": 256, "Frames": 10000, "Pixelsize": 130},
          {"Generated by": "my_analysis_script.py"},
      ]
      io.validate_locs(locs, info)  # raises io.LocsFormatError if not loadable
      io.save_locs("my_locs.hdf5", locs, info)  # writes the format marker

.. _files-format-versions:

Format versions
~~~~~~~~~~~~~~~

Since Picasso 1.0, every file Picasso writes records the version of its
format: localization files (also pick properties and identifications) in two
attributes of the HDF5 file (attributes of its root, ``/``),
``picasso_format``, e.g. ``"1.0"``, and ``picasso_version``, the version of
Picasso that wrote the file; YAML files (pick regions, calibrations, masks,
nanoTRON models) in the key ``picasso_format``; drift ``.txt`` files in a
header line ``# picasso_format 1.0``. The format version ``MAJOR.MINOR`` has
two parts:

- The **minor** version grows with additions that older versions of Picasso
  can ignore, e.g., a new optional column or dataset. Picasso opens files of a
  newer minor version.
- The **major** version grows when the meaning of existing content changes,
  e.g., the unit of a column. Picasso refuses files of a newer major version
  and asks to be updated.

Older versions of Picasso ignore the attributes and open current files as
before.

Files without a format version were written before Picasso 1.0 (or by other
software following the old convention): their lengths are in camera pixels.
Picasso opens them and brings them to the current format, see
:ref:`files-legacy`: the lengths are converted to nm with the camera pixel
size, and the column ``likelihood`` (the log-likelihood of MLE fits, named so
before Picasso 0.11) is read as ``log_likelihood``. Dark times (column
``dark``) computed before Picasso 0.11.3 were one frame longer than now;
Picasso warns when the metadata of a file shows that its dark times are of
that kind, or, for files written before Picasso 1.0, does not say.

.. list-table:: Versions of the localization file format.
   :name: files-format-version-table
   :widths: 15, 20, 65
   :header-rows: 1

   * - Format
     - Picasso
     - Changes
   * - 1.0
     - 1.0.0
     - First versioned format: the ``picasso_format`` and ``picasso_version``
       attributes. **Every length is in nm** (``x``, ``y``, ``lpx``, ``lpy``,
       ``sx``, ``sy``, pick positions, drift, ...; before 1.0 they were in
       camera pixels, ``z`` in nm). ``Pixelsize`` is required in the
       metadata. Steps of Picasso record ``Picasso version`` and ``Step`` in
       the metadata (see :ref:`files-metadata-settings`).

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``io.FORMAT_VERSION`` is the version that ``io.save_locs`` (and every
   other writer) records.

   .. code-block:: python

      from picasso import io

      io.get_format_version("movie_locs.hdf5")  # (1, 0); None before Picasso 1.0
      io.FORMAT_VERSION  # "1.0"

.. _files-legacy:

Files written before Picasso 1.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   **Updating to Picasso 1.0 needs no action for your data.** Every file
   written by an earlier version of Picasso opens in 1.0 as before, in every
   module, on the command line and in Python: localizations (also picked,
   linked and clustered ones, cluster centers and pick properties), pick
   regions, drift files, astigmatism, spline and lateral calibrations, SPINNA
   masks, nanoTRON models, and Render's field of view and view settings
   files. Picasso converts them to nm as it reads them, and opening a file
   never changes it. Converting files on disk (see below) is optional.

Until Picasso 1.0, the coordinates ``x`` and ``y``, the precisions ``lpx``
and ``lpy``, the spot widths ``sx`` and ``sy`` and everything derived from
them (pick positions, drift, cluster centers, pick properties, ...) were stored
in camera pixels, while ``z`` was in nm. Since 1.0, every length is in nm.
Picasso tells the two apart by the format marker (see
:ref:`files-format-versions`): a file without it was written before 1.0.

**How old files are read.** Whenever Picasso opens such a file (Render,
Filter, Localize, the command line, ``picasso.io.load_locs``), it converts the
lengths to nm in memory with the camera pixel size: the ``Pixelsize`` of the
metadata, or, if the file does not record it, the value you are asked for (the
GUIs ask, the command line and the Python functions take ``pixelsize``).
Localize records ``Pixelsize`` in the metadata, so files localized with
Picasso are usually converted without a question. The conversion is recorded
as a ``Convert units`` step in the metadata, and the file is written in nm,
with the format marker, when it is saved again. Pick regions and drift files
are converted the same way when they are loaded. The other direction does not
work: earlier versions of Picasso read files saved by 1.0 as camera pixels, so
whoever opens your new files needs Picasso 1.0 or later as well.

**Converting files once** rewrites files written before Picasso 1.0 (in camera
pixels) in the units convention of Picasso 1.0 (nm, with the format marker).
This saves the question for the pixel size and makes the files readable as nm
by other software. Files that are in nm already are skipped.

- In Render, ``File > Convert files to nm...`` takes any number of pre-1.0
  files (localizations and their ``.yaml``, pick regions, drift ``.txt``,
  astigmatism and lateral calibrations, SPINNA masks, nanoTRON models), asks
  for the camera pixel size where a file does not record it, and writes the
  converted files next to the originals with ``_nm`` appended to their names
  (or replaces the originals, if you tick that option), see
  :ref:`render-convert-files`.
- On the command line, ``picasso convert-units FILE [FILE ...]``
  (``--pixelsize`` for files that do not record it, ``--in-place`` to replace
  the originals), see :ref:`cmd-convert-units`.
- In Python, ``picasso.legacy.convert_file`` and
  ``picasso.legacy.convert_files``; the in-memory conversions
  (``legacy.locs_to_nm``, ``legacy.picks_to_nm``, ...) are there as well.

What is converted: the columns in :ref:`Table 1 <files-localization-columns>`
and :ref:`Table 2 <files-molecular-maps>` that were in camera pixels (and
their ``_mean``/``_std`` statistics in pick properties), the Filter ranges of
those columns in the metadata, the positions of pick regions, the ``x`` and
``y`` drift, the polynomial
coefficients (spot widths) and lateral corrections of calibrations, the
extent of SPINNA masks, and the pick diameter and rendered pixel size of
nanoTRON models. Counts of camera pixels that are not lengths (``Width``,
``Height``, box sizes, regions of interest) stay as they are.

.. note::

   A file written in nm by other software must carry the format marker
   (``picasso_format = "1.0"``, see :ref:`files-matlab-export` for MATLAB and
   ``picasso.io.save_locs`` for Python); without it, Picasso takes its lengths
   for camera pixels and multiplies them by the pixel size.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      from picasso import io, legacy

      # loading converts in memory; the pixel size is needed only when the
      # metadata of an old file does not record it
      locs, info = io.load_locs("old_locs.hdf5")
      locs, info = io.load_locs("older_locs.hdf5", pixelsize=130)
      picks, shape, size = io.load_picks("old_locs_picks.yaml", pixelsize=130)

      # converting files
      legacy.convert_file("old_locs.hdf5")  # -> old_locs_nm.hdf5 (+ .yaml)
      legacy.convert_file("old_locs_drift.txt", pixelsize=130, in_place=True)
      legacy.convert_files(paths, pixelsize=130)  # (path, output, message)



Importing HDF5 files in MATLAB and Origin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In MATLAB, execute the command:

.. code-block:: matlab

   locs = h5read(filename, dataset)

Replace ``dataset`` with ``/locs`` for localization files and with ``/groups``
for pick property files.

In Origin, select ``File > Import > HDF5`` or drag and drop the file into the main window.

.. _files-matlab-export:

Saving localizations from MATLAB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function below saves localizations from MATLAB so that Picasso can open
them (see :ref:`files-minimum-requirements`): the table ``/locs`` in an HDF5
file with the format marker, and the metadata in a YAML file of the same
name. ``locs`` is a struct of column vectors ``frame`` (counting from 0),
``x``, ``y``, ``lpx`` and ``lpy``, all in nm; ``pixelsize`` is the camera
pixel size in nm, which sets the size of the field of view in camera pixels.
For example, ``save_picasso('C:/data/my_locs.hdf5', locs, 130)``.

.. code-block:: matlab

   function save_picasso(path, locs, pixelsize)
   % the table /locs: frame as uint32, the other columns as double
   names = {'frame', 'x', 'y', 'lpx', 'lpy'};
   table_type = H5T.create('H5T_COMPOUND', 4 + 4 * 8);
   H5T.insert(table_type, 'frame', 0, H5T.copy('H5T_NATIVE_UINT32'));
   data.frame = uint32(locs.frame(:));
   for i = 2:5
       H5T.insert(table_type, names{i}, 4 + (i - 2) * 8, ...
                  H5T.copy('H5T_NATIVE_DOUBLE'));
       data.(names{i}) = double(locs.(names{i})(:));
   end
   file = H5F.create(path, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
   space = H5S.create_simple(1, numel(data.frame), []);
   dataset = H5D.create(file, '/locs', table_type, space, 'H5P_DEFAULT');
   H5D.write(dataset, table_type, 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', data);
   H5D.close(dataset);
   H5S.close(space);
   H5T.close(table_type);
   % the format marker: the lengths are in nm (Picasso 1.0)
   string_type = H5T.copy('H5T_C_S1');
   H5T.set_size(string_type, 3);
   scalar = H5S.create('H5S_SCALAR');
   marker = H5A.create(file, 'picasso_format', string_type, scalar, ...
                       'H5P_DEFAULT');
   H5A.write(marker, string_type, '1.0');
   H5A.close(marker);
   H5S.close(scalar);
   H5T.close(string_type);
   H5F.close(file);

   % the metadata: frames count from 0; Width and Height are camera pixels,
   % and localizations at x >= Width * Pixelsize or y >= Height * Pixelsize
   % are discarded (better still, use the camera's size)
   [folder, name] = fileparts(path);
   yaml = fopen(fullfile(folder, [name '.yaml']), 'w');
   fprintf(yaml, 'Frames: %d\n', max(data.frame) + 1);
   fprintf(yaml, 'Width: %d\n', floor(max(data.x) / pixelsize) + 1);
   fprintf(yaml, 'Height: %d\n', floor(max(data.y) / pixelsize) + 1);
   fprintf(yaml, 'Pixelsize: %g\n', pixelsize);
   fprintf(yaml, 'Generated by: MATLAB\n');
   fclose(yaml);
   end

.. _files-localization-hdf5:

Localization HDF5 Files
-----------------------

``locs, info = picasso.io.load_locs`` reads the localizations together with
their metadata (see :ref:`files-metadata-settings` below).

The localization table is stored as a dataset of the HDF5 file in the path
``/locs``. This table can be explored by opening the HDF5 file with
:doc:`filter`. The localization table can have an unlimited number of columns.
Table 1 explains the main column names in Picasso; the required columns
are in bold (``z`` only for 3D data, see :ref:`files-minimum-requirements`).

.. csv-table:: Table 1: Name, description and data type for the main columns used in Picasso.
   :name: files-localization-columns
   :file: table01.csv
   :widths: 15, 65, 20
   :header-rows: 1

The required columns are listed in :ref:`files-minimum-requirements`.

.. _files-molecular-maps:

Molecular maps (cluster centers) HDF5 Files
-------------------------------------------

Molecular maps generated with RESI or by single-protein resolution imaging
(i.e., generated by SMLM clusterer or G5M) can be opened in
``Picasso: Render`` just like localizations. The column names change slightly
to reflect the different data type. Table 2 explains the main column names in
molecular maps.

.. csv-table:: Table 2: Name, description and data type for the main columns used in molecular maps in Picasso.
   :file: table02.csv
   :widths: 15, 65, 20
   :header-rows: 1

.. _files-pick-properties:

HDF5 Pick Property Files
------------------------

When selecting ``File > Save pick properties`` in ``Picasso: Render``, the
properties of picked regions are stored in an HDF5 file. Within the HDF5 file,
the data table is stored in the path ``/groups``.

Each row in the ``groups`` table corresponds to one picked region (see
:ref:`render-save-pick-properties` for how it is calculated). The
localizations of a pick are first linked into binding events; Table 3 lists
the columns.

These files are the basis for qPAINT, which counts binding sites from the
binding kinetics (`Jungmann et al., Nature Methods, 2016 <https://doi.org/10.1038/nmeth.3804>`__): each row holds the pick's mean dark time, its qPAINT
index (``qpaint_idx_cdf``) and the number of binding sites (``n_units``). The
influx rate that ``n_units`` is computed from is calibrated in Render's Info
dialog, which also shows these values averaged over the picks (see
:ref:`render-info-picks`).

.. csv-table:: Table 3: Name, description and data type for the columns of pick property files.
   :name: files-pick-property-columns
   :file: table03.csv
   :widths: 15, 65, 20
   :header-rows: 1

Bright and dark times are counted in frames:

- The bright time (``len``) of a binding event runs from its first to its last
  frame, both included.
- The dark time is the number of frames without signal between the end of one
  binding event and the start of the next one in the same pick.
- Dark times are only measured between binding events: the frames before the
  first and after the last binding event in a pick are not counted, and a pick
  with a single binding event has no dark time.

.. important::

   Before Picasso 0.11.3, the dark time was the difference between the first
   frame of a binding event and the last frame of the previous one, i.e., one
   frame longer than now. Dark times, qPAINT influx rates and numbers of units
   calculated with earlier versions are therefore not directly comparable;
   recalibrate the influx rate with the current version. See
   ``picasso.postprocess.dark_times`` for the details.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   See :ref:`render-save-pick-properties` for computing them from Python.

   .. code-block:: python

      import pandas as pd

      groups = pd.read_hdf("movie_locs_pickprops.hdf5", key="groups")
      print(groups[["group", "n_events", "dark_mean", "n_units"]])


.. _spatial-index:

Spatial index
-------------
``picasso.io.save_locs`` also stores the spatial index that
``Picasso: Render`` uses for fast zooming, rotating and picking (a
multi-resolution grid over the localizations, see ``picasso.spatial_index``)
in the HDF5 group ``/render_index``, for files of 100,000 localizations or
more (``render_index=True`` or ``False`` overrides this). When such a file is
opened in Render, the stored index is read instead of built.

Before it is used, the index is checked against the localizations (every
entry must cover exactly one row and every grid cell must hold only the rows
that fall in it), so a file whose ``/locs`` table was rewritten by other
software or by a script without ``save_locs`` is simply re-indexed; the check
cannot be fooled by a stale index. Older Picasso versions and other software
ignore the group.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      io.save_locs("movie_locs.hdf5", locs, info, render_index=False)


.. _files-dtypes:

Data types
----------

Localizations are stored with float32 floating-point columns and a uint32
``frame`` column, whatever dtypes a pipeline produced in memory
(``picasso.lib.standardize_dtypes``, applied when loading, saving and by most
processing functions). float32 resolves far below any localization precision,
and halves memory and file size compared to float64.

.. _files-metadata-settings:

Metadata
--------

Metadata describes a dataset (a movie, localizations, identifications,
spots or pick properties): the size of the field of view, the number of frames, the
pixel size, the processing history, etc. In Picasso, metadata is represented
internally as a list of dictionaries (``info``) where each step of analysis
appends a new dictionary to the list. The keys that localizations need are
listed in :ref:`files-minimum-requirements`; example files can be found in the
`samples/data folder <https://github.com/jungmannlab/picasso/tree/master/samples/data>`_.

Picasso stores the metadata of an HDF5 file in two places:

- embedded in the file, as a JSON string in the dataset ``/metadata``, so the
  file is self-contained and keeps its metadata when it is moved or renamed;
- by default, also in a YAML file with the same name and the extension
  ``.yaml``, which is easy to inspect and edit.

When loading (``picasso.io.load_info``, used by ``load_locs``), Picasso reads
the YAML file if there is one and the embedded ``/metadata`` otherwise; if
neither exists, a ``NoMetadataFileError`` is raised. The YAML file thus takes
precedence and can be edited to change the metadata, or deleted, except for
files saved by Picasso before v0.11, which have no embedded metadata. A YAML
file next to a file that Picasso saves is always overwritten, also with
``Save metadata in .yaml`` off, so that an outdated one cannot hide the new
metadata.

Each processing step of Picasso appends a dictionary that starts with
``Generated by`` (e.g., ``Picasso v1.0.0 Filter``, shown in the metadata
viewers), ``Picasso version`` and ``Step`` (e.g., ``Filter``); the remaining
keys hold the parameters of the step.

Two user settings in ``~/.picasso/settings.yaml`` (see
:ref:`user-settings-file`; also available under ``File > Picasso settings`` in
any module) control what is written:

``Save metadata in .yaml`` (default: ``true``)
   Set to ``False`` to save only the embedded ``/metadata`` dataset, without
   the YAML file (an existing YAML file is still overwritten, see above).
``Save Micro-Manager metadata`` (default: ``true``)
   Movies acquired with MicroManager carry a large block of microscope
   properties, which Picasso reads from the movie (as
   ``Micro-Manager Metadata``) and copies into the metadata of the
   localizations fitted from it. Set this setting to ``false`` to leave it
   out. This applies when localizing only - the block is dropped as the movie
   metadata is carried over into the localizations (or saved
   identifications/spots).

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``get_from_metadata`` searches the list from the last entry to the
   first. Picasso's functions that return new localizations also return
   their metadata with the step appended (see :ref:`api-locs-info`). Record
   your own processing step with ``lib.append_metadata``, which returns a new
   list; ``lib.step_info`` makes the dictionaries of Picasso's own steps.

   .. code-block:: python

      from picasso import io, lib

      info = io.load_info("movie_locs.yaml")             # or the .hdf5 path
      pixelsize = lib.get_from_metadata(info, "Pixelsize")

      info = lib.append_metadata(
          info, {"Generated by": "my_analysis_script.py", "Threshold": 0.5}
      )
      io.save_locs("movie_locs_processed.hdf5", locs, info)
      io.save_info("movie_locs_processed.yaml", info)   # the YAML alone


.. _files-yaml:

YAML Files
----------

YAML files are document-oriented text files that can be opened and changed
with any text editor. Picasso uses them for:

- the metadata of movies, localizations, identifications, spots and pick
  properties (see :ref:`files-metadata-settings` above);
- the display settings of exported images and animations, saved next to the
  image or video (see :ref:`render-export-images` and
  :ref:`render-animation`);
- pick regions (see :ref:`render-save-pick-regions`);
- 3D astigmatism calibrations (see :ref:`localize-3d-calibration`);
- SPINNA structures and masks (see :ref:`spinna-structures-tab` and
  :ref:`spinna-mask-generation-tab`);
- the parameters of simulations (:doc:`simulate`), designs (:doc:`design`)
  and Nanotron models (:doc:`nanotron`);
- the user settings (see :ref:`user-settings-file`).

Picasso reads every YAML file safely: no code named in a file is run. Before
version 1.0, Picasso wrote numpy numbers, tuples and arrays with
Python-specific tags (e.g., ``!!python/object/apply:numpy...``); their values
are rebuilt from the data in the file. Since 1.0, metadata and calibrations are
written as plain YAML that any YAML reader can read, and the files that hold
lengths (pick regions, calibrations, masks, nanoTRON models) carry the format
marker ``picasso_format: '1.0'``, which says that the lengths are in nm (see
:ref:`files-legacy`). To read a Picasso YAML file in a script, use
``picasso.io.load_yaml``, which also reads the files written before 1.0;
``yaml.safe_load`` refuses their Python-specific tags.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   .. code-block:: python

      from picasso import io

      info = io.load_info("movie_locs.yaml")
      picks, shape, size = io.load_picks("movie_locs_picks.yaml")  # nm
      calibration = io.load_calibration("beads_3d_calib.yaml")
      mask_info = io.load_yaml("mask.yaml")  # any other Picasso YAML file
      io.save_yaml("my_file.yaml", {"My key": 1.0})  # plain YAML
