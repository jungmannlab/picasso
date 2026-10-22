.. _render-spt:

Single-Particle Tracking
========================

.. note::

   Single-particle tracking is an experimental feature. Bug reports and
   suggestions for improvement are much appreciated on our `GitHub issues
   page <https://github.com/jungmannlab/picasso/issues>`__.

Single-particle tracking (SPT) follows individual molecules over time to
measure how they move. Picasso provides a basic SPT workflow, following
Module 7 of `Martens, Turkowyd and Endesfelder, Frontiers in Bioinformatics,
2021 <https://doi.org/10.3389/fbinf.2021.817254>`__:

1. **Track particles**: link the localizations of consecutive frames into
   tracks by nearest-neighbor tracking (:ref:`render-spt-track`).
2. **Show the tracks** as lines through their localizations
   (:ref:`render-spt-overlay`).
3. **Analyze the tracks**: measure diffusion coefficients from the mean
   squared displacement and from the distribution of jump distances, and
   summarize every track (:ref:`render-spt-analysis`).

Both steps are found in ``Postprocess > Single-particle tracking`` and are
also available as the commands :ref:`picasso track <cmd-track>` and
:ref:`picasso diffusion <cmd-diffusion>` and in the Python module
``picasso.spt``.

Tracking and analysis are 2D: z is ignored. The tracking has no motion model
and does not merge or split tracks, so it is meant for sparse particles, i.e.,
when particles are further apart than they move between two frames.

.. _render-spt-workflow:

Before tracking
---------------

- **Localize** the movie as usual (see :doc:`../localize`). For moving
  particles, a robust detection matters more than the best precision.
- **Drift correction**: RCC and AIM assume a static sample and cannot be used
  for moving particles. Correct drift with fiducial markers instead
  (``Postprocess > Undrift from picked``, see :ref:`render-marker-drift`).
- **Regions**, e.g., cells: pick them and save the picked localizations (see
  :doc:`picking`). The ``group`` column then holds the pick of every
  localization, and tracks never link localizations of different picks.
- **Do not link** the localizations (``Postprocess > Link localizations``):
  linking merges the localizations of consecutive frames into one, which
  removes the time course that tracking needs. Linked channels are skipped.

.. _render-spt-track:

Tracking particles
------------------

Open ``Postprocess > Single-particle tracking > Track particles...``. With
several channels loaded, choose one or *Apply to all sequentially*.

``Max. jump (nm)`` (500 nm by default)
   Maximum distance between consecutive localizations of a track. It should
   be larger than the jumps of the fastest particles and smaller than the
   distance between particles. For free diffusion in 2D, the root mean square
   jump is :math:`\sqrt{4 D \Delta t}`, so about three times that covers
   nearly all jumps: e.g., 850 nm for :math:`D = 1\,\mu m^2/s` and a frame
   time of 20 ms.
``Max. gap (frames)`` (1 by default)
   Maximum number of frames a track may skip, e.g., when the fluorophore
   blinks or the particle is briefly not detected. 0 links consecutive frames
   only.
``Min. track length`` (2 by default)
   Minimum number of localizations of a track; the localizations of shorter
   tracks are removed. Fast particles leave the focus sooner, so a large
   minimum favors slow particles.
``Frame time (ms)``
   Time between the starts of two consecutive frames (not the exposure
   time). It is not used for tracking, but stored with the tracks for their
   analysis. Picasso remembers the last value used.

The tracking goes through the frames in order. Every localization continues
the track whose last localization is the closest one within ``Max. jump``,
looking back up to ``Max. gap`` skipped frames. A track gets at most one
localization per frame and a localization belongs to one track: when several
tracks reach for the same localization (or one track for several), the closest
pair is linked first, then the next closest, and so on. Of two equally close
tracks, the one that skipped fewer frames wins. Localizations that do not
continue a track start a new one.

The tracked localizations replace the channel (save them with
``File > Save localizations``). The track id of every localization is stored
in the ``group`` column, numbered in the order of the tracks' first frames; a
previous ``group`` column (e.g., the picks) is kept as ``group_input``. For a
single channel, the localizations are therefore colored by track (see
:ref:`render-coloring`). The parameters are added to the metadata. Tracking
a channel that is already tracked is not possible; reload the original
localizations to track them again with other parameters.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``max_jump`` is in nm. The frame time is optional and stored in the
   metadata for the analysis.

   .. code-block:: python

      from picasso import io, spt

      locs, info = io.load_locs("movie_locs.hdf5")

      tracked, info = spt.link_tracks(
          locs, info,
          max_jump=500,                 # nm
          max_gap=1,                    # frames a track may skip
          min_length=2,                 # localizations per track
          frame_time=20,                # ms, stored in the metadata
      )
      print(tracked["group"].nunique(), "tracks")
      io.save_locs("movie_locs_tracks.hdf5", tracked, info)

.. _render-spt-overlay:

Showing tracks
--------------

Tick ``Tracks`` in ``View > Display settings`` (see
:ref:`render-display-settings`) to draw every track of the tracked channels
as a line through its localizations, in the order of their frames. It is
ticked automatically after tracking.

- ``Color``: ``Auto`` (default) draws each track in the color of its
  localizations: one of eight colors per track for a single channel, the
  channel colors for several channels. Any other color is used for all
  tracks. ``Line``, ``Width`` and ``Opacity`` set the look of the lines.
- ``Max. tracks drawn`` (10,000 by default): when more tracks are in view,
  only every k-th track is drawn, so that panning and zooming stay smooth. The
  same tracks are drawn while you pan. The line below it tells how many
  tracks are in view and drawn.

The tracks are part of ``Export current view``, ``Export complete image`` and
``Export view manually``. Their settings, except whether they are shown, are
saved under ``Tracks`` in the ``Render`` section of
``~/.picasso/settings.yaml`` (see :ref:`user-settings-file`). The tracks are
not drawn in the 3D view.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``track_segments`` gives the line segments between consecutive
   localizations of every track; ``select_track_segments`` culls them to the
   field of view.

   .. code-block:: python

      from picasso import render, spt

      segments, track_id, rows = spt.track_segments(tracked)
      mask, n_visible, n_drawn = render.select_track_segments(
          segments, track_id, viewport, max_tracks=10_000
      )
      # qimage shows the rendered viewport ((y_min, x_min), (y_max, x_max))
      qimage = render.draw_tracks(
          qimage, viewport, segments[mask],
          style=render.OverlayStyle(color="yellow", line_width=1.5),
      )

.. _render-spt-analysis:

Analyzing tracks
----------------

Open ``Postprocess > Single-particle tracking > Analyze tracks (MSD, jump
distances)...`` and choose a tracked channel. The dialog asks for:

- ``Frame time (ms)``, taken from the tracking if it was given there.
- **Mean squared displacement**: ``Max. lag (frames)`` (10 by default), the
  largest time lag of the MSD curve, and ``Fit points`` (4 by default), the
  number of time lags fitted (see :ref:`render-spt-msd`).
- **Jump distances**: the ``Localization error`` of the fit and whether to
  ``Correct for the max. jump distance`` (see :ref:`render-spt-jump-distances`).

The results open in a window with a summary and three plots: the ensemble MSD
with its fit, the histogram of the jump distances with the fit of 1, 2 or 3
populations (chosen in the toolbar; the one with the lowest BIC is shown
first), and the histogram of the track lengths. ``Save results...`` saves them
(see :ref:`render-spt-track-table`).

The analysis follows Module 7 of `Martens, Turkowyd and Endesfelder, Frontiers
in Bioinformatics, 2021 <https://doi.org/10.3389/fbinf.2021.817254>`__.
Please cite the paper when using these analyses.

.. _render-spt-msd:

Mean squared displacement
~~~~~~~~~~~~~~~~~~~~~~~~~

For a time lag of :math:`n` frames, the squared distances between all pairs
of localizations of a track that are exactly :math:`n` frames apart are
averaged. Pairs that would include a skipped frame are left out. The ensemble
MSD averages over all pairs of all tracks; its error bars are the standard
error, with the tracks as independent units.

For free diffusion in 2D,

.. math::

   \mathrm{MSD}(\tau) = 4 D \tau + 4 \sigma^2,

where :math:`\tau` is the time lag, :math:`D` the diffusion coefficient and
:math:`\sigma` the localization error. A straight line is fitted to the first
``Fit points`` lags, which gives :math:`D` from the slope and :math:`\sigma`
from the intercept. The first lags are the most precise, since they have the
most pairs. The intercept also contains the motion blur during the camera
exposure, which reduces it, so :math:`\sigma` is an *apparent* localization
error; it is not given if the intercept is negative. The shape of the MSD
curve tells the kind of motion: it bends down for confined and up for directed
motion.

The same fit gives the diffusion coefficient of every track (column
``D_um2_s`` of the track summary). It needs a pair of localizations at each of
the fitted lags, i.e., at least ``Fit points`` + 1 localizations, and is
noisy for short tracks.

.. _render-spt-jump-distances:

Jump distances
~~~~~~~~~~~~~~

The distances between localizations of a track in consecutive frames (jumps
across skipped frames are left out) are fitted with a mixture of 1, 2 and 3
freely diffusing populations. In 2D, the jump distance :math:`r` of a
population with the diffusion coefficient :math:`D` follows

.. math::

   p(r) = \frac{2 r}{s} \exp\left(-\frac{r^2}{s}\right),
   \qquad s = 4 D \Delta t + 4 \sigma^2,

with the frame time :math:`\Delta t`. The fit gives the diffusion coefficient
and the fraction of the jumps of every population. Picasso fits the jump
distances themselves by maximum likelihood, so the result does not depend on
a histogram binning; the paper above fits the histogram instead, and the
histogram is only shown for comparison. Compare the numbers of populations by
the BIC (Bayesian information criterion): lower is better, and a further
population is only justified if it lowers the BIC clearly.

``Localization error``
   One time lag cannot tell the localization error from diffusion, since both
   broaden the distribution in the same way, so :math:`\sigma` is fixed.
   ``None`` (default) gives *apparent* diffusion coefficients,
   :math:`D + \sigma^2 / \Delta t`. ``Median localization precision`` uses
   the median of ``lpx`` and ``lpy``; note that this is a lower bound of the
   real error of moving particles. ``Custom`` uses the value given, e.g.,
   measured on immobilized particles.
``Correct for the max. jump distance``
   Jumps longer than ``Max. jump`` could not be linked and are missing, which
   biases the diffusion coefficients of fast populations low when ``Max.
   jump`` is close to their typical jumps. If ticked, the fit accounts for
   the cut-off, and the fractions refer to all jumps, including the missing
   ones. This is not part of the original protocol and is off by default.

The fractions count jumps, not molecules: a molecule seen for many frames
contributes many jumps, and fast molecules leave the focus sooner.

.. _render-spt-track-table:

Saved results
~~~~~~~~~~~~~

``Save results...`` asks for a file name (``<name>_track_stats.csv`` by
default) and saves three files:

``<name>_track_stats.csv``
   One row per track:

   .. list-table::
      :header-rows: 1
      :widths: 25 75

      * - Column
        - Description
      * - ``group``
        - Track id.
      * - ``group_input``
        - Group (e.g., pick) of the track before tracking, if any.
      * - ``n_locs``
        - Number of localizations.
      * - ``first_frame``, ``last_frame``
        - First and last frame.
      * - ``n_frames``
        - Frames from the first to the last, skipped frames included.
      * - ``duration_s``
        - Time from the first to the last localization (s).
      * - ``x_mean``, ``y_mean``
        - Mean position (nm).
      * - ``n_jumps``
        - Number of jumps between consecutive frames.
      * - ``mean_jump_nm``
        - Mean jump distance (nm); empty without jumps.
      * - ``D_um2_s``
        - Diffusion coefficient from the MSD of the track (µm²/s); empty if
          the track is too short.

``<name>_msd.csv``
   The ensemble MSD: time lag in frames and seconds, MSD and its standard
   error (µm²), and the numbers of pairs and tracks per lag.
``<name>_diffusion.yaml``
   The parameters, the MSD fit and the fits of the jump distances.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``analyze_tracks`` runs the whole analysis; the steps are also available
   one by one (``msd``, ``fit_msd``, ``jump_distances``,
   ``fit_jump_distances`` and ``track_table``). Diffusion coefficients are in
   µm²/s, MSDs in µm², and jump distances and localization errors in nm.

   .. code-block:: python

      from picasso import io, spt

      tracked, info = io.load_locs("movie_locs_tracks.hdf5")
      result = spt.analyze_tracks(
          tracked, info,
          frame_time=20,              # ms; None: from the metadata
          max_lag=10,                 # frames
          n_fit=4,                    # lags fitted
          sigma=0,                    # nm, or "precision"
          n_populations=(1, 2, 3),
      )
      print(result["msd_fit"]["D"])                   # um^2/s
      fit = result["jd_fits"][2]
      print(fit["D"], fit["fractions"], fit["bic"])
      spt.save_analysis("movie_locs_tracks", result)  # CSV and YAML files

      # the steps one by one
      jumps = spt.jump_distances(tracked)              # nm
      fit = spt.fit_jump_distances(jumps, frame_time=20, n_populations=2)
      fig = spt.plot_jump_distances(jumps, fit)
      fig.savefig("jump_distances.png")
