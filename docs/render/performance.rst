.. _render-performance:

Performance
===========

Two settings in the ``Render`` section of the user settings file (see
:ref:`user-settings-file`) control the speed of ``Picasso: Render``: how many
CPU threads it may use (:ref:`render-cpu-usage`) and whether it renders on the
graphics card (:ref:`render-gpu-rendering`).

.. _render-cpu-usage:

CPU usage on shared workstations
--------------------------------

Rendering uses a limited number of CPU worker threads so that Picasso stays
polite on shared analysis computers where several users work at the same
time. The budget is set in the user settings file ``~/.picasso/settings.yaml``
(also editable via ``File > Picasso settings`` in any module):

.. code-block:: yaml

    Render:
      cpu_utilization: 0.5
      max_workers: 4
      interaction_subsample: auto
      max_blur_width: 100
      gpu:
        vram_budget_mb: 8192

``cpu_utilization``
   The fraction of CPU cores that rendering may use, a number between 0 and 1
   (exclusive). The default is 0.5. Invalid values silently fall back to the default.
``max_workers`` (optional)
   An absolute cap on the number of worker threads; it wins over
   ``cpu_utilization``. For example, set it to 4 on a 64-core workstation to
   leave the remaining cores to your colleagues regardless of the fraction.
   Remove the key to disable the cap.
``interaction_subsample``
   Controls the live previews shown *while* you pan or zoom: during a gesture,
   Picasso renders a subset of localizations (with the contrast compensated,
   so brightness does not change) and follows up with the full-quality image a
   moment after the gesture pauses.

   - ``auto`` (the default): a preview shows 500,000 localizations or a tenth
     of the localizations in view, whichever is larger. For example, with
     20 million localizations in view, the preview shows 2 million. Fields of
     view with fewer than 500,000 localizations are always rendered in full.
     The tenth keeps faint structures of large datasets visible while you
     move.
   - An integer sets a fixed target instead.
   - ``0`` or ``off`` disables previews so every frame renders at full
     quality.
``max_blur_width`` (nm)
   Applies to the two blur methods that use each localization's own precision
   (*Individual loc. prec.* and its isotropic variant): localizations whose
   ``lpx`` or ``lpy`` exceeds this value are **not rendered** at all.

   - Unfiltered data occasionally contains localizations with absurd
     precisions of hundreds of screen pixels — their Gaussian would spread a
     negligible intensity over a large FOV, yet drawing one is
     computationally expensive.
   - The default is 100 nm. ``0`` or ``off`` renders everything regardless of
     precision.
   - We recommend removing such localizations upstream in Picasso: Filter.
``gpu``
   Described in :ref:`render-gpu-rendering`.

The settings file is read every time a render starts, so changes apply
immediately, without restarting Picasso. At least one worker is always used
and, on Windows, the number of workers is capped at 61 (a limitation of
Python's process handling).

Picasso: Render writes the ``Render`` keys it does not find in the file with
their defaults when it starts (``max_workers`` excepted, as it is optional), so
every setting is visible and editable. If an edit breaks the file (e.g., a
stray tab), Picasso keeps the broken copy, warns you and uses the defaults;
the previous version is always kept as ``settings.yaml.bak``. See
:ref:`user-settings-file`.

.. _render-gpu-rendering:

GPU rendering
-------------

Localizations can be rendered on the graphics card instead of the CPU, which
makes large multiplexed datasets interactive: the whole dataset is uploaded to
the GPU once and every view afterwards is computed there, typically several
times faster than the CPU worker threads, with the sharp image arriving where
the CPU path shows a preview.

It works on any recent graphics card through the ``wgpu`` package, which the
one-click installers include; for ``pip`` installations, see
:doc:`/getting-started/installation`.

The rendering is controlled by the ``gpu`` section of the ``Render`` settings
shown above:

.. code-block:: yaml

    Render:
      gpu:
        enabled: auto
        adapter: high-performance
        vram_budget_mb: 8192

``enabled``
   - ``auto`` (the default) renders on the GPU whenever one can be initialized
     and silently uses the CPU otherwise.
   - ``on`` does the same but records a warning in the
     :ref:`error log <error-log>` when the GPU cannot be used, for
     troubleshooting.
   - ``off`` never touches the GPU.

   Whatever the setting, a problem on the GPU never interrupts your work: the
   affected image is simply rendered on the CPU. Renders of fewer than
   20,000 localizations always use the CPU, which is faster for them. In the
   3D view the threshold is 1,000 localizations.
   ``View > Show info`` shows which renderer is in use (see
   :ref:`render-info-display`), and when the last render was handed from the
   GPU to the CPU it names the reason (the full traceback is in the
   :ref:`error log <error-log>`).
``adapter``
   Which graphics card to use on computers with several, e.g. laptops with an
   integrated and a dedicated GPU. ``high-performance`` (the default) asks the
   system for the dedicated one, ``low-power`` for the integrated one; any
   other text selects the first adapter whose name contains it, e.g.
   ``NVIDIA`` or ``Intel``. The chosen adapter is recorded in the log.
``vram_budget_mb``
   Caps the GPU memory (in MB) the uploaded localizations may occupy; the
   default is 8192 (8 GB).

   - When the cap is reached, the least recently rendered channels are
     released, and a single channel larger than the cap is rendered in pieces
     instead of failing.
   - ``0`` removes the cap.
   - As a rule of thumb, a two-dimensional dataset needs 16 bytes per
     localization (about 1 GB for 60 million localizations), a
     three-dimensional one with per-localization angles up to 28 bytes.

.. _render-gpu-requirements:

Requirements
~~~~~~~~~~~~

- A graphics card from any manufacturer (NVIDIA, AMD, Intel or Apple), with
  a current driver that supports Metal (macOS 10.13 or later, all Apple
  silicon Macs), Direct3D 12 (Windows 10 or later) or Vulkan (Linux, with the
  vendor's Vulkan driver installed). Unlike other GPU functionalities in Picasso, GPU
  rendering does not strictly require an NVIDIA card or CUDA. Integrated graphics work
  as well, but dedicated cards are faster.
- The ``wgpu`` package: the one-click installers ship it, pip users install
  ``picassosr[wgpu]``.

.. _render-gpu-troubleshooting:

When Show info says CPU
~~~~~~~~~~~~~~~~~~~~~~~

When ``View > Show info`` says CPU, check in this order:

1. ``Render: gpu: enabled`` in ``File > Picasso settings`` is not set to
   ``off``.
2. The dataset is not too small: renders of fewer than 20,000 localizations
   (1,000 in the 3D view) always use the CPU, and the info dialog reports the renderer of the last
   render.
3. ``wgpu`` is installed in the environment that runs Picasso
   (``pip install picassosr[wgpu]``); the one-click installers include it.
4. Set ``enabled: on``, restart Picasso, open the data again and read
   the :ref:`error log <error-log>`: the line *GPU rendering unavailable* names
   the reason (no adapter found, a driver too old for Direct3D 12 or Vulkan, a
   name given under ``adapter`` that matches no card).
5. On a computer with several GPUs, set ``adapter`` to (part of) the name of
   the card you want, e.g. ``NVIDIA``.

.. _render-gpu-details:

How the GPU path compares to the CPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- All blur methods render on the GPU, in 2D and in the 3D view, except the
  :ref:`adaptive histogram and the jittered triangulation <render-blur>`,
  which always render on the CPU.
- Zoomed-in views use the same spatial index as the CPU path: once the field
  of view covers less than a tenth of the image, only the localizations around
  it are handed to the GPU, so rendering cost follows what is visible.
- The live previews during panning and zooming work exactly as on the CPU
  (``interaction_subsample``, counted over the localizations in the visible
  field of view) and are drawn straight from the localizations already
  resident on the GPU; the sharp image follows a moment after the gesture
  pauses.
- GPU and CPU images agree to well within display precision: raw intensities
  match to about a thousandth of the image maximum, and histogram counts are
  exact save for a localization that sits within floating-point rounding of a
  pixel edge.
- Repeated GPU renders can differ in the last decimal places, since the GPU
  sums in a hardware-dependent order. Set ``enabled: off`` if you need
  bit-for-bit reproducible images (this also applies to ``picasso.render``).
