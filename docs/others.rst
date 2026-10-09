Settings and Customization
==========================

.. _sound-notifications:

Sound notifications
-------------------
Picasso supports sound notifications for processes running longer than 1
minute. In Localize, Render and SPINNA, these can be selected in the ``File`` menu in
the menu bar.

- The available files are read from ``~/.picasso/notification_sounds``.
  ``.mp3`` and ``.wav`` files are supported.
- The default sound notification is saved automatically when manually
  changed, under ``filename`` in the ``Sound_notification`` section of
  ``~/.picasso/settings.yaml`` (see :ref:`user-settings-file`); the default is
  no sound (``None``).

.. _custom-notifications:

Custom notifications
~~~~~~~~~~~~~~~~~~~~
To add custom notification sounds, copy the sound files (``.mp3`` or
``.wav``) to ``~/.picasso/notification_sounds`` in your home directory.
Picasso creates the folder the first time it looks for sounds. The location
is the same for every installation type (one-click installer, PyPI or a
cloned repository) and is kept when Picasso is uninstalled or updated.

.. _appearance:

Appearance
----------
``File > Appearance...`` in any module sets the look of the Picasso windows:

- **Theme**: *System* (default) is light or dark like the operating system and follows it when it changes; *Light* and *Dark* fix it; *Native* is the platform's own PyQt style, the look of Picasso before version 0.12.
- **Accent color**: the color of selections, checked and default buttons, sliders and focus frames; a preset or any color (*Custom...*).
- **Font size**: the size of the text in percent of the system's.
- **Density**: *Compact* reduces the spacing of the controls, e.g., for small laptop screens.
- **Toolbar**: how the toolbars of ``Picasso: Render``, ``Picasso: Localize``, ``Picasso: Filter`` and ``Picasso: Average`` show their buttons: *Icons and text* (default), *Icons*, *Text* or *Hidden*. Which actions they hold is set in :ref:`toolbars`.
- **Menus**: *Show icons in menus* (default on) shows the icons next to the actions of the menus.

Changes apply immediately to the module that is open and are saved in the
``Appearance`` section of ``~/.picasso/settings.yaml`` (see
:ref:`user-settings-file`); other modules take them on when they are started
next.

The theme does not change the colors of the image content (rendered
localizations, picks); the background and labels of Design's canvas follow
it. Charts (e.g., Filter's histograms, Render's plot windows, Simulate's
previews) are light or dark like the windows unless another theme is chosen in
their plot settings.

.. _toolbars:

Toolbars
--------
The toolbars of ``Picasso: Render``, ``Picasso: Localize``,
``Picasso: Filter`` and ``Picasso: Average`` hold actions of the menus, by
default the most used ones, e.g., opening and saving, the tools of Render and
identifying and fitting in Localize. They can be dragged by their handle to
any side of the window; the side is remembered for the next start.

*Customize toolbar...* in ``File > Appearance...``, or a right-click on the
toolbar, chooses the buttons. The left side lists the actions of the menus,
grouped by menu, with a search field; the right side lists the toolbar's
buttons from left to right (or top to bottom).

- **Add** (or a double-click) puts the actions selected on the left after the selected button; actions that are on the toolbar already are grayed out.
- **Remove** takes the selected button off the toolbar.
- **Separator** adds a separator after the selected button.
- **Up** and **Down**, or dragging, change the order.
- **Label** gives the selected button a shorter text than its action's, e.g., *RCC* for *Undrift by RCC*; empty for the action's own.
- **Icon** gives it another icon, or *No icon* to show only its label. The
  menus show the chosen icon too.

  *Choose file...* takes an icon of your own: an SVG, or a PNG or ICO with a
  transparent background. Only its shape is used, drawn in the colors of the
  theme like Picasso's icons, so single-color line icons (e.g., from
  `Lucide <https://lucide.dev>`_) fit best. The file is copied to
  ``~/.picasso/icons``; images put in that folder are offered too.
- **Restore Defaults** shows the default buttons again, after asking for confirmation.

*OK* applies the toolbar to every open window of the module and saves it in
the ``Toolbars`` section of ``~/.picasso/settings.yaml`` (see
:ref:`user-settings-file`). Whether the buttons show icons, text or both is
set in :ref:`appearance`.

Plugin actions (see :doc:`plugins`) can be on the toolbar too; a button whose
action is not in the menus, e.g., of a plugin that is disabled, is kept and
shown in italics in the dialog, and returns with its plugin.

.. _user-settings-file:

User settings file
------------------
Picasso keeps its user settings in ``~/.picasso/settings.yaml``
(``C:\Users\<you>\.picasso\settings.yaml`` on Windows): the last directory
used, the Render colormap, the Localize parameters, the CPU and GPU budgets of
rendering, the sound notification, and so on. Each module owns a section of
the file (``Render``, ``Localize``, ...).

The file can be edited with any text editor or via ``File > Picasso settings``
in any module, and changes apply the next time the setting is read - for most
settings immediately, without restarting Picasso. The search field above the
editor (:kbd:`Ctrl+F`) highlights every occurrence of a text, e.g., a
setting's name; :kbd:`Enter` and :kbd:`Shift+Enter` step through them.

A setting that is missing from the file is written into it with its default
the first time it is needed, so every setting a module uses is visible and
editable in the file; ``Picasso: Render``, for example, writes all of its
``Render`` keys when it starts. Optional keys that are off unless present
(such as ``Render: max_workers``) are the exception.

Every module loads the file, changes its own keys and writes the whole file back, so the file is guarded against mistakes:

- before it is rewritten, the previous version is kept as
  ``settings.yaml.bak``, so the last good version is always at hand;
- a file that cannot be parsed (a stray tab or a misplaced colon is enough) is
  never overwritten silently: a copy is kept as ``settings.yaml.broken``, a
  warning goes to the :ref:`error log <error-log>` and default settings are
  used - ``Picasso: Render`` also tells you so when it starts.

.. tip::

   To get your settings back after a parsing error, fix the YAML in the kept
   ``settings.yaml.broken`` copy and paste it into
   ``File > Picasso settings``, which validates the YAML before saving.

Reference: every key
~~~~~~~~~~~~~~~~~~~~
The tables below list every key Picasso reads from or writes to ``settings.yaml``, grouped by the section (top-level, or a module) that owns it.

Top level
^^^^^^^^^

.. list-table::
   :widths: 22 10 68
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``Save metadata in .yaml``
     - ``True``
     - Also write a sidecar ``.yaml`` metadata file next to saved localizations, in addition to the metadata already embedded in the ``.hdf5`` file. See :ref:`files-metadata-settings`.
   * - ``Save picks in metadata``
     - ``False``
     - Embed the picked regions (shape, size, positions) in the metadata when saving picked localizations from Render. See :ref:`render-save-picks-in-metadata`.
   * - ``Save Micro-Manager metadata``
     - ``True``
     - Keep the (often large) MicroManager property block when copying a movie's metadata into the localizations fitted from it. See :ref:`files-metadata-settings`.

``Sound_notification``
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 22 10 68
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``filename``
     - ``None`` (no sound)
     - The sound file (from ``~/.picasso/notification_sounds``) played on long-running jobs in Localize, Render and SPINNA. See :ref:`sound-notifications` above.

``Appearance``
^^^^^^^^^^^^^^

.. list-table::
   :widths: 22 10 68
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``mode``
     - ``System``
     - Theme of the windows: ``System``, ``Light``, ``Dark`` or ``Native``. See :ref:`appearance`.
   * - ``accent``
     - ``#2A78D6``
     - Accent color as a hexadecimal code.
   * - ``font_scale``
     - ``100``
     - Font size in percent of the system's, 30 to 300.
   * - ``density``
     - ``Comfortable``
     - Spacing of the controls: ``Comfortable`` or ``Compact``.
   * - ``toolbar``
     - ``Icons and text``
     - Buttons of the toolbars of Render, Localize, Filter and Average: ``Icons``, ``Icons and text``, ``Text`` or ``Hidden``.
   * - ``menu_icons``
     - ``True``
     - Show the icons of the actions in the menus.

``Toolbars``
^^^^^^^^^^^^

.. list-table::
   :widths: 22 10 68
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``Render toolbar``, ``Localize toolbar``, ``Filter toolbar``, ``Average toolbar``
     - not set (the default buttons, on top)
     - The toolbar of the module (see :ref:`toolbars`), with the keys:

       - ``items``: the buttons from left to right, as the paths of the menu
         actions, e.g., ``File > Open``, and ``---`` for a separator;
       - ``labels`` and ``icons``: the custom label and icon of a button by
         its path; an icon is the name of one of Picasso's,
         ``user:<file name without extension>`` for one in
         ``~/.picasso/icons``, or ``none``;
       - ``area``: the side of the window, ``Top``, ``Bottom``, ``Left`` or
         ``Right``.

``Render``
^^^^^^^^^^

.. list-table::
   :widths: 22 10 68
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``Colormap``
     - ``magma`` (GUI), ``viridis`` (command line)
     - The last colormap selected for the loaded channels, restored on the next Render session or ``picasso render`` run. See :ref:`render-colormap-setting`.
   * - ``Colormap Property``
     - ``gist_rainbow``
     - The colormap used when rendering by property. Kept separately from ``Colormap`` above.
   * - ``CustomColormaps``
     - *(none)*
     - User-defined colormaps created with the custom colormap editor, keyed by name; each is a list of color stops. See :ref:`render-colormap-setting`.
   * - ``PWD``
     - *(last used)*
     - Remembered last value: the directory used in Render's file dialogs.
   * - ``Colorbar format``
     - ``.png``
     - Format of the exported colorbar/LUT image next to a "render by property" export - ``.png`` or ``.svg``. See :ref:`render-colorbar-format`.
   * - ``ToolStyles``
     - *(last used)*
     - Remembered last value: the appearance of the picks, the Measure tool and the Move tool's shift label, set in Render's ``Tools > Tools Settings``. Saved when Render is closed.
   * - ``cpu_utilization``
     - ``0.5``
     - Fraction of CPU cores rendering's worker pool may use. See :ref:`render-cpu-usage`.
   * - ``max_workers``
     - *(unset = no cap)*
     - Absolute cap on rendering worker threads, overriding ``cpu_utilization``. See :ref:`render-cpu-usage`.
   * - ``interaction_subsample``
     - ``auto``
     - Target number of localizations rendered during live pan/zoom previews. See :ref:`render-cpu-usage`.
   * - ``max_blur_width``
     - ``100`` (nm)
     - Localizations whose precision (``lpx``/``lpy``) exceeds this are skipped by the per-localization blur methods. See :ref:`render-cpu-usage`.
   * - ``gpu: enabled``
     - ``auto``
     - Whether/when rendering uses the GPU backend - ``auto``, ``on`` or ``off``. See :ref:`render-gpu-rendering`.
   * - ``gpu: adapter``
     - ``high-performance``
     - Which GPU to use on a computer with several. See :ref:`render-gpu-rendering`.
   * - ``gpu: vram_budget_mb``
     - ``8192``
     - GPU memory budget (MB) for resident localization uploads; ``0`` removes the cap. See :ref:`render-gpu-rendering`.

``Localize``
^^^^^^^^^^^^

.. list-table::
   :widths: 22 10 68
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``cpu_utilization``
     - ``0.8``
     - Fraction of CPU cores used by the spot identification/fitting worker pool. See :ref:`localize-cpu-fitting`.
   * - ``PWD``
     - *(last used)*
     - Remembered last value: the directory used in Localize's file dialogs.
   * - ``box_size``
     - *(last used)*
     - Remembered last value: the ``Box side length`` in the ``Parameters`` dialog.
   * - ``gradient``
     - *(last used)*
     - Remembered last value: the ``Min. net gradient`` in the ``Parameters`` dialog.
   * - ``temporal_median``
     - *(last used)*
     - Remembered last value: the temporal median filter window.
   * - ``temporal_median_on``
     - *(last used)*
     - Remembered last value: whether the temporal median filter was ticked.
   * - ``gaussian_filter_sigma``
     - *(last used)*
     - Remembered last value: the Gaussian pre-filter sigma.
   * - ``identification_method``
     - *(last used)*
     - Remembered last value: the identification ``Method`` (net gradient or B-spline wavelet).
   * - ``wavelet_threshold``
     - *(last used)*
     - Remembered last value: the ``Wavelet threshold``. See :ref:`localize-wavelet`.
   * - ``wavelet_noise``
     - *(last used)*
     - Remembered last value: the wavelet ``Noise estimate``.
   * - ``wavelet_min_area``
     - *(last used)*
     - Remembered last value: the wavelet ``Min. region area``.
   * - ``fit_model``
     - *(last used)*
     - Remembered last value: the PSF fit **Model**.
   * - ``fit_optimizer``
     - *(last used)*
     - Remembered last value: the fit **Optimizer** (least squares / MLE).
   * - ``fit_mode``
     - *(last used)*
     - Remembered last value: the **Fit mode** selection.
   * - ``Columns to save``
     - *(all columns)*
     - Which localization columns are ticked in ``File`` > ``Select columns to save...`` when saving fit results.

``Filter``
^^^^^^^^^^

.. list-table::
   :widths: 22 10 68
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``PWD``
     - *(last used)*
     - Remembered last value: the directory used in Filter's file dialogs.

``SPINNA``
^^^^^^^^^^

.. list-table::
   :widths: 22 10 68
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``PWD``
     - current working directory
     - Remembered last value: the directory used in SPINNA's file dialogs.
   * - ``Fitting mode``
     - *(last used)*
     - The fitting method (``bayesian``, ``coarse to fine`` or ``brute force``) chosen in the "Optional settings" dialog. See :doc:`spinna`.
   * - ``NND fonts``
     - *(widget defaults)*
     - Font family and size for the title, axis labels and ticks of the NND plot, set in *Plot settings*. See :doc:`spinna`.

``Updates``
^^^^^^^^^^^
Written and read only by the update-notification feature, not by any module's own GUI.

.. list-table::
   :widths: 22 10 68
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``Last update check``
     - *(unset)*
     - Timestamp (ISO format) of the last update check; a new check is due again 24 hours after it.
   * - ``Skipped version``
     - *(unset)*
     - A release version the user chose "skip this version" for; suppresses notifications for that version only.
   * - ``Snoozed until``
     - *(unset)*
     - Timestamp (ISO format) until which "remind me later" suppresses notifications.
   * - ``Disabled``
     - ``False``
     - Update checks are switched off entirely.
