.. _faq:

FAQ and troubleshooting
=======================

Answers to the questions asked most often, with links to the details. If your
problem is not here, search the `GitHub issues
<https://github.com/jungmannlab/picasso/issues?q=is%3Aissue>`__ or open a new
one with the log file attached, see :ref:`faq-reporting`.

Installing and starting
-----------------------

.. _faq-other-smlm:

Can I use Picasso for STORM, PALM or other SMLM data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. Picasso started as the analysis software for DNA-PAINT, and DNA-PAINT is
still where many new features come from, but the core workflow is not specific
to it: Localize fits any movie of blinking single molecules, and Filter,
Render, Average and SPINNA work with localizations from any SMLM technique,
including drift correction, clustering, 3D and multi-channel analysis.
Localizations from other software can be imported, see
:ref:`faq-import`.

A few tools build on DNA-PAINT and are of little use otherwise: Design
(DNA origami docking sites), Simulate (DNA-PAINT acquisitions) and the
kinetics-based analyses such as qPAINT.

Which installer should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Windows:** the one-click installer from the `release page
  <https://github.com/jungmannlab/picasso/releases>`__. Its *CUDA* edition
  adds GPU fitting in Localize and supports plugins that rely on CUDA
  and needs an NVIDIA graphics card; GPU
  rendering in Render works with both editions. The editions install to
  ``C:\Picasso`` and ``C:\Picasso-CUDA`` and can be installed side by side.
- **macOS:** the one-click installer runs on Apple Silicon (M1 and later)
  only. On an Intel Mac, install Picasso with pip.
- **Linux, scripts and notebooks:** ``pip install picassosr`` in an
  environment with Python 3.11 to 3.14 (not 3.15 yet, see
  :ref:`faq-python-315`).

See :doc:`getting-started/installation`.

macOS says it cannot check Picasso for malicious software
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Picasso is distributed without an Apple Developer ID. Click the ``?`` button
in the warning and follow the instructions to open Picasso anyway (`Apple's
guide <https://support.apple.com/en-gb/guide/mac-help/mh40616/mac>`__), once
for the main Picasso app and once for each module app you use, e.g., Picasso
Render. The main app must stay in the *Applications* folder: the module apps
start it from there and otherwise say *Picasso.app not found in Applications
folder*.

Render or Localize does not start for some Windows users
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If Picasso is installed in *Program Files*, users without administrator rights
may not be able to start Render and Localize. Install it outside *Program
Files*, e.g., in the default ``C:\Picasso``. Windows Defender and SmartScreen
may ask for permission several times while you download and install it.

Picasso does not start after ``pip install``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install Picasso into a new environment with Python 3.11 to 3.14, e.g.,
``conda create -n picasso python=3.13``, then ``pip install picassosr``:
packages of other software in the same environment can conflict with
Picasso's.

.. _faq-python-315:

Why is Python 3.15 not supported yet?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Picasso's own code runs on Python 3.15, but it cannot be installed there with
pip yet, and part of it would not work:

- **Dependencies without Python 3.15 packages.** h5py, numexpr, ndindex and
  PyYAML do not publish ready-made packages (wheels) for Python 3.15 yet, so
  pip tries to build them from their source code. That needs a C/C++ compiler
  and, for h5py, the HDF5 library, and usually fails with:

  .. code-block:: text

     Failed to build installable wheels for some pyproject.toml based projects
     ╰─> h5py, numexpr, ndindex

- **No GPU fitting.** CUDA code, such as GPU fitting in Localize, runs on
  numba-cuda, which NVIDIA only maintains now (new development moved to
  numba-cuda-mlir) and which will not support Python 3.15.

Moreover, ``pip install picassosr`` with Python 3.15 falls back to Picasso
0.8.8, the last version that did not declare which Python versions it
supports, and then fails to build its outdated dependencies (e.g., NumPy
1.24).

Use Python 3.14 (``conda create -n picasso python=3.14``) or the one-click
installer. Picasso will support Python 3.15 once these packages are available
for it.

Which version of Picasso am I running?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The title bar of every window shows it, e.g., *Picasso v1.0.0: Render*. In
Python, ``import picasso; print(picasso.__version__)``.

.. _faq-update:

How do I update Picasso?
~~~~~~~~~~~~~~~~~~~~~~~~

Picasso checks for a new version at most once a day and tells you about it;
``Updates: Disabled: true`` in the :ref:`user settings <user-settings-file>`
switches the checks off. Your settings, camera configuration and plugins are
kept when you update.

With the one-click installers (from v1.0 on), click *Install Now* in the
notification. Picasso downloads the installer of the new version, checks it
against the checksum published with the release, closes, installs the update
and opens again. Other Picasso programs have to be closed first, and on
Windows the installer asks for administrator rights, as when installing by
hand. On macOS, ``Picasso.app`` and the module apps in the same folder (e.g.,
*Picasso Render*) are replaced; if that is not possible, e.g., without the
rights to change the Applications folder, the disk image opens to install it
by hand. If the update was not installed, Picasso says so when it opens again;
the log is in ``~/.picasso/logs/update.log``.

You can also download and install the new version from the
`release page <https://github.com/jungmannlab/picasso/releases>`__ yourself;
on Windows, there is no need to uninstall the old version first, except for
the *GPU* edition of versions before 1.0: its successor, the *CUDA* edition,
installs beside it, so uninstall *Picasso-GPU* (Windows *Settings* > *Apps*).
With pip, ``pip install --upgrade picassosr``.

Updating from a version before 1.0? See :doc:`migration`.

Files and units
---------------

Why are my coordinates about 100 times larger since Picasso 1.0?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since Picasso 1.0, every length is in nm instead of camera pixels: ``x``,
``y``, ``lpx``, ``lpy``, ``sx``, ``sy``, pick positions and sizes, drift, and
the distances you give on the command line and in Python (e.g.,
``picasso link -d`` or the radius of ``picasso dbscan``). A distance given in
camera pixels where nm are expected is about 100 times too small.
:doc:`migration` lists what changed and the replacement of each setting,
command and function.

Do I need to convert my old files?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No. Picasso 1.0 opens every file written by earlier versions and converts its
lengths to nm as it reads it, with the ``Pixelsize`` of its metadata; the file
on disk is not changed. If a file does not record the pixel size, the windows
ask for it, and the Python functions take it as ``pixelsize=``; on the
command line, convert such files once with
``picasso convert-units --pixelsize 130 FILES``. Otherwise, converting the
files on disk is optional, see :ref:`files-legacy`.

Astigmatism (3D) and lateral calibrations written before 1.0 are converted
with the pixel size of the movie or localizations they are used with, which
assumes they were recorded with the same camera.

Can colleagues with an older Picasso open my files?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No. Files written by Picasso 1.0 are in nm, and earlier versions read them as
camera pixels, so everyone who opens them needs Picasso 1.0 or later. Their
files open in Picasso 1.0, though.

Picasso says that localizations were not loaded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Picasso leaves out localizations with missing or infinite values, with
negative values (e.g., coordinates or precisions) and outside the field of
view, and says how many. If most of them lie beyond the field of view, either

- the file is in nm but lacks the ``picasso_format`` attribute, so Picasso
  took it for camera pixels and multiplied it by the pixel size: files written
  in nm by other software need the attribute, see the note in
  :ref:`files-legacy`;
- or ``Width``, ``Height`` or ``Pixelsize`` in its metadata are too small.

Picasso says that the path is too long
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On Windows, a full path, e.g.,
``C:\Users\me\Documents\experiments\...\movie_locs_render_picked.hdf5``, may
have at most 259 characters, and each step of Picasso that saves a file adds
to its name (``_locs``, ``_render``, ``_picked``, ...). Move the files to a
folder closer to the drive, e.g., ``C:\data``, or shorten the folder and file
names. Alternatively, an administrator can `enable long paths in Windows
<https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation>`__.
Even then, a single folder or file name may have at most 255 characters.

.. _faq-import:

How do I open localizations from other software?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Render's ``File > Open`` imports ThunderSTORM ``.csv``, SMAP ``_sml.mat`` and
PyME ``.h5r`` and ``.hdf`` files, see :ref:`render-open`; on the command line,
``picasso csv2hdf``, ``smap2hdf`` and ``pyme2hdf`` convert them. To write files
for Picasso from your own code, see :ref:`files-minimum-requirements`: they
need the columns ``frame``, ``x``, ``y``, ``lpx`` and ``lpy`` (in nm), the
metadata keys ``Width``, ``Height``, ``Frames`` and ``Pixelsize``, and the
format marker, which ``picasso.io.save_locs`` adds.

How do I read Picasso files in Python or MATLAB?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Python, ``locs, info = picasso.io.load_locs(path)`` returns the
localizations as a pandas DataFrame and the metadata as a list of
dictionaries; ``pandas.read_hdf(path, key="locs")`` reads the localizations
without Picasso. In MATLAB, ``h5read(path, '/locs')``. See :ref:`files-hdf5`
and :ref:`files-matlab-export`.

To save localizations from Python, always use
``picasso.io.save_locs(path, locs, info)``, not ``DataFrame.to_hdf`` or
``h5py``: only ``save_locs`` writes the format marker and the metadata that
Picasso needs to read the file correctly, see
:ref:`files-minimum-requirements`.

How do I export localizations to other software?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``File > Export localizations`` in Render writes them for ThunderSTORM, SMAP,
PyME, ImageJ and other software, see :ref:`render-export-localizations`; on
the command line, e.g., ``picasso hdf2csv``. Since Picasso 1.0, the exports
are in nm.

Localize
--------

Which movie formats can Localize open?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TIFF stacks of any kind (plain ``.tif`` and ``.tiff``, OME-TIFF, NDTiffStack,
BigTIFF, also files larger than 4 GB), ``.nd2``, ``.ims``, ``.stk``, ``.lsm``
and ``.raw``; ``.czi`` and ``.lif`` with ``pip install "picassosr[czi]"`` and
``"picassosr[lif]"`` (Python 3.12 or later; included in the one-click
installers). TIFF files do not need to be renamed to ``.ome.tif`` or converted
to ``.raw``. The frames must be grayscale, one per page. For a movie split into
several μManager files, open the first one. See :ref:`localize-file-formats`.

What do I enter for EM gain, baseline, sensitivity and pixel size?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

They convert camera counts into photons and thus decide the photon counts and
the localization precision:

- ``EM Gain``: the electron-multiplying gain; 1 for cameras without EM gain,
  such as sCMOS cameras.
- ``Baseline``: the camera's offset, i.e., the average count of a frame
  recorded without light.
- ``Sensitivity``: electrons per count for the readout mode you recorded
  with, from the camera's specification sheet.
- ``Pixel size (nm)``: the size of a camera pixel in the sample, i.e.,
  divided by the magnification.

Localize movies as the camera recorded them, not background-subtracted or
rescaled (e.g., to 8 bit): the conversion needs the original counts. To have
Picasso fill in these values for your camera, set up the camera configuration,
``~/.picasso/config.yaml`` (Localize: ``File > Open camera config file
location``), see :ref:`localize-camera-config`.

I localized a movie with the wrong pixel size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Localize it again with the correct ``Pixel size (nm)`` (``-px`` on the command
line). The pixel size converts the fitted positions, precisions and spot widths
from camera pixels to nm while the movie is localized, and the ``z`` of
astigmatic 3D data is fitted from the spot widths, so changing ``Pixelsize`` in
the metadata afterwards does not correct them; it only changes the size of
the field of view. Localize takes the pixel size from the movie's metadata or from the
``Pixelsize`` of the camera in the camera configuration, so set it there once
(see :ref:`localize-camera-config`).

Localize does not select my camera or calibration automatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Picasso selects them from the metadata of movies recorded with μManager and of
``.nd2`` files; for other files, select them by hand. The names in the camera
configuration must match the metadata character for character, see
:ref:`localize-camera-config-matching`. ``File > Show metadata...`` shows the
names a movie carries.

How do I choose the box size and the minimum net gradient?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a simple 2D setup, a box side length of 7 should work (6σ + 1 for a PSF with σ of about
0.9 camera pixels); the box must hold the whole spot, but a much larger box
is more likely to take in its neighbors. Tick ``Preview`` and raise ``Min. net gradient``
until only spots are marked, no background. The net gradient is not a pixel
value: it grows with the brightness and sharpness of a spot and with the box
size, so it can exceed the brightest pixel. See :ref:`Box size and net gradient <localize-box-size>` and
:ref:`first-steps-localize`.

How do I remove bad localizations?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Filter them after localizing, in Picasso: Filter or with ``picasso filter``,
e.g., by localization precision, photons, spot width or the goodness of the
fit, ``reduced_chi_square`` (about 1 for a good fit). Conditions such as
``lpx < 10 and photons > 500`` can be typed in, see :ref:`filter-expression`.

``Use GPU`` is missing or grayed out
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU fitting needs an NVIDIA graphics card and either the CUDA edition of the
Windows installer or ``pip install "picassosr[gpu]"``; without them, the option
is hidden. It is grayed out for fitting methods that have no GPU version.
There is no GPU fitting on a Mac. See :ref:`localize-gpu`. GPU rendering in
Render is independent of this and works with any graphics card.

How do I calibrate astigmatic 3D imaging?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Record a z-stack of beads, calibrate it in Localize and load the calibration
for your measurements, see :ref:`localize-3d-calibration` and
:ref:`localize-fitting-z`. The ``Magnification factor`` (0.79 by default)
corrects ``z`` for the refractive-index mismatch between the immersion medium
and the sample. For multichannel and biplane data, see :doc:`localize/spline`.

Localize keeps the computer busy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Localize: cpu_utilization`` in the user settings sets the fraction of the
CPU cores that fitting uses, see :ref:`localize-cpu-fitting`.

Render
------

Which drift correction should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start with AIM (``Postprocess > Undrift by AIM``): it needs no markers and is
fast. If it fails, increase ``Segmentation``: segments that are too short do not
overlap. You can follow it with *Undrift from picked* on
fiducials or on many copies of the same structure. RCC needs no markers either
but is slower and corrects only x and y. See :ref:`render-drift-correction`.

Every correction saves the drift in a ``_drift.txt`` file. To apply it to
another channel, use ``Postprocess > Apply drift from an external file`` (see
:ref:`render-drift-tools`) or ``picasso undrift --from-file``.

``View > Show info`` says Render uses the CPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Datasets of fewer than 20,000 localizations (1,000 in the 3D view) are always
rendered on the CPU. Otherwise, check that GPU rendering is enabled in the user
settings and read the line *GPU rendering unavailable* in the
:ref:`log file <error-log>` (in the terminal, if you started Picasso from one),
which names the reason. On a computer with several graphics cards, choose one
with ``adapter``. See :ref:`render-gpu-troubleshooting`.

Render slows down a shared computer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Render: cpu_utilization`` (0.5 by default) sets the fraction of the CPU
cores rendering may use, see :ref:`render-cpu-usage`.

HDBSCAN fails with a cluster selection epsilon above 0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a bug of scikit-learn with NumPy 2.4 and later. Set the epsilon to 0,
or install an older NumPy, ``pip install "numpy<2.4"``. The one-click
installers ship NumPy 2.3 and are not affected.

Where are the areas and other properties of my clusters?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tick ``Save cluster centers`` in the DBSCAN, HDBSCAN or SMLM clusterer dialog:
a ``_centers.hdf5`` file next to the clustered localizations then holds one
row per cluster, with its mean frame, size, number of localizations, area and
more (in nm), see :ref:`Clustering <render-clustering-output>`. It opens in
Render and Filter like localizations, and ``picasso cluster-filter`` keeps the
clustered localizations whose clusters have a property within given limits.

Command line and scripts
------------------------

The ``picasso`` command is not found
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **pip:** activate the environment Picasso is installed in.
- **Windows one-click installer:** the installer adds Picasso to the
  ``PATH``; open a new terminal after installing.
- **macOS one-click installer:** ``picasso`` is not on the ``PATH``; add it
  with a link in ``/usr/local/bin``, see :ref:`installation
  <installation-command-line>`, or run
  ``/Applications/Picasso.app/Contents/MacOS/picasso``.
- **Linux:** ``release/create_linux_shortcuts.py`` puts ``picasso`` into
  ``~/.local/bin``, see :doc:`getting-started/installation`.

How do I process many files at once?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``picasso localize FOLDER`` localizes every movie in the folder, in all the
formats Localize opens; a movie split over several ``.tif`` files counts once.
For astigmatic 3D, pass ``-zc`` and ``-mf``, or the command asks for them. Most other commands take a file name pattern in
quotes, e.g., ``picasso link "*_locs.hdf5"``. See :doc:`cmd`.

A command says it is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Picasso 1.0 renamed some commands and options to one style, e.g.,
``pair-correlation`` for ``pc`` and ``--r-max`` for ``--rmax``. The old names
keep working until Picasso 2.0, see :ref:`cmd-renamed`. Distances are in nm
since 1.0, see :ref:`migration-cli`.

My script no longer prints progress or messages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions that take a while show a progress bar with ``progress="console"``,
and Picasso reports through Python's :mod:`logging`; ``logging.basicConfig(
level=logging.INFO)`` shows its messages. See :doc:`api/index`.

A script shows a ``FutureWarning``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The script uses a name that is deprecated and will be removed; the warning
names the replacement and the version that removes it. See
:ref:`versioning-deprecations`. Names that Picasso 1.0 removed are listed with
their replacements in :doc:`migration`.

Picasso Server
--------------

``picasso server`` says it needs Streamlit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Picasso Server is an optional extra of the Python package:
``pip install "picassosr[server]"``. The one-click installers include it.

Other computers cannot open Picasso Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Picasso Server only accepts connections from the computer it runs on. Start
it with ``picasso server --network`` to allow other computers; it has no login,
so do this only in a trusted network. See :ref:`server-browser-app`.

The watcher ignores my files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The watcher processes ``.raw``, ``.ome.tif`` and ``.ims`` movies, see
:ref:`server-watcher`.

Plugins
-------

A plugin I copied into the plugins folder does nothing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plugins are found but not run until you enable them: under
``Plugins > Browse online plugins...`` or with
``picasso plugins enable <file.py>``. See :ref:`plugins-manual-install`.

An enabled plugin has no menu entry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A plugin appears only in the module it was written for. If it fails to load,
``picasso plugins list`` shows the error. After fixing the plugin, use
``Plugins > Reload plugins``. See :ref:`plugins-without-gui`.

Settings and reporting problems
-------------------------------

Where does Picasso keep its files?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the folder ``.picasso`` in your home folder (e.g.,
``C:\Users\<you>\.picasso`` on Windows), shared by all installations and kept
when Picasso is updated or removed:

- ``settings.yaml``: the user settings, see :ref:`user-settings-file`;
- ``config.yaml``: the camera configuration, which you create, see
  :ref:`localize-camera-config`;
- ``logs/picasso.log``: the log file, see :ref:`error-log`;
- ``plugins/``: the plugins;
- ``notification_sounds/``: the sounds played when long jobs finish;
- ``app_0410.db``: the database of localized files that Picasso Server reads.

How do I reset Picasso's settings?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Close all Picasso windows and rename or delete ``~/.picasso/settings.yaml``;
Picasso starts with the default settings. A settings file that cannot be read
is not overwritten: Picasso uses the defaults and keeps a copy as
``settings.yaml.broken``, see :ref:`user-settings-file`.

.. _faq-reporting:

Picasso shows an error or crashes: what should I send?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open an issue on `GitHub <https://github.com/jungmannlab/picasso/issues>`__
with what you did, the version of Picasso and your operating system, and
attach the log file ``~/.picasso/logs/picasso.log``, which holds the
tracebacks of all errors. ``Show Details...`` in the error message shows the
traceback of that error. See :ref:`error-log`.
