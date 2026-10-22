============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

.. contents:: Table of Contents
   :local:
   :depth: 2

Types of Contributions
----------------------

You can contribute in many ways:

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/jungmannlab/picasso/issues.

Many problems are answered in the `FAQ <https://picassosr.readthedocs.io/en/latest/faq.html>`__. If you are reporting a bug, please include:

* The version of Picasso (shown in the title bar of every window) and how you installed it (one-click installer or pip).
* Your operating system name and version.
* The log file ``~/.picasso/logs/picasso.log``, which holds the tracebacks of all errors (see `Reporting problems <https://picassosr.readthedocs.io/en/latest/getting-started/installation.html#error-log>`__).
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Picasso could always use more documentation, whether as part of the official Picasso docs, in docstrings, or even on the web in blog posts, articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/jungmannlab/picasso/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions are welcome :)

Contributing Code
-----------------

Development Setup
~~~~~~~~~~~~~~~~~

Here's how to set up ``picasso`` for local development. If you are new to git and GitHub, you might want to familiarize yourself with the basics of git and GitHub first. The internet is full of great resources on this topic!

1. Fork the ``picasso`` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/picasso.git

3. Follow the instructions in the *Developer installation* tab of the `installation guide <https://picassosr.readthedocs.io/en/latest/getting-started/installation.html>`__.

4. Set up ``pre-commit`` hooks. This ensures your code is automatically formatted with ``black`` before each commit::

    $ pre-commit install

5. (Optionally) Build the docs (Note that you need to install sphinx and the rtd-theme)::

    $ cd docs/
    $ make html  # docs will be found in _build/html/. Open `index.html` to view the docs; macOS users can do `open _build/html/index.html` to open the docs in your default browser.

Development Workflow
~~~~~~~~~~~~~~~~~~~~

1. Create a branch for local development. Use a descriptive name prefixed with the type of change, e.g. ``fix/broken-rendering``, ``feat/new-filter``, ``docs/update-readme``::

    $ git checkout -b fix/broken-rendering

   Now you can make your changes locally.

2. Keep your fork in sync with the upstream repository to avoid merge conflicts::

    $ git remote add upstream https://github.com/jungmannlab/picasso.git
    $ git fetch upstream
    $ git rebase upstream/main

   You only need to run the ``git remote add`` command once.

Code Style
^^^^^^^^^^

This project uses `black <https://github.com/psf/black>`_ for code formatting with a line length of 79 characters. The ``pre-commit`` hook (set up above) will automatically format your code on each commit. You can also run it manually::

    $ black --line-length=79 .

Testing
^^^^^^^

3. When you're done making changes, run the tests from the repository root::

    $ pytest

   To run a specific test file for faster iteration::

    $ pytest tests/test_render.py

   To see which code the tests run (with ``pytest-cov`` installed; CI shows the same table on the summary page of the Linux, Python 3.14 job)::

    $ pytest --cov --cov-report=term-missing

Committing & Pushing
^^^^^^^^^^^^^^^^^^^^^

4. Commit your changes using the imperative mood (e.g. "Fix bug" not "Fixed bug" or "Fixes bug"). Reference related issues in the commit message (#123 in the example below)::

    $ git add .
    $ git commit -m "Fix rendering crash for large datasets (#123)"
    $ git push origin fix/broken-rendering

5. Update the changelog (``changelog.md``) with a brief summary of your changes under the appropriate section.

Changing Public Names
^^^^^^^^^^^^^^^^^^^^^

The functions and classes on the Python API pages (each module's ``__all__``), the commands and options of the command line and the file formats are public: see the stability policy in ``docs/versioning.rst``. Do not remove or rename them; deprecate them with ``picasso.deprecation`` instead, so that they keep working for two more minor releases. ``tests/test_stability.py`` compares the public names with snapshots in ``tests/data/stability``; after a deliberate change, record it with the command below, which refuses to record the removal of a name that was not deprecated first::

    $ python tests/regen_stability_snapshots.py

Functions on Localizations
^^^^^^^^^^^^^^^^^^^^^^^^^^

A public function on localizations takes them and their metadata first, ``f(locs, info, *, ...)`` (lists for several channels, ``f(locs, infos, *, ...)``), and every other argument after a ``*``, so that it is passed by name. Writers take the path first, ``f(path, locs, info, *, ...)``. Make ``info`` optional (``info: list[dict] | None = None``) if the function does not need it.

A function that returns new localizations (a processing step) returns ``locs, info`` first and further results after them. ``info`` is the complete metadata of the result: the input metadata with one entry recording the step appended by ``lib.append_metadata``, which copies the entries, so that the caller's metadata is not modified::

    def my_step(
        locs: pd.DataFrame,
        info: list[dict] | None = None,
        *,
        radius: float = 10.0,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """...

        Returns
        -------
        locs : pd.DataFrame
            ...
        info : list of dicts
            Metadata of ``locs``: ``info`` with an entry recording the
            step appended, see ``lib.append_metadata``.
        """
        ...
        entry = lib.step_info("My step", **{"Radius (nm)": radius})
        return locs, lib.append_metadata(info, entry)

Use ``lib.step_info`` for the entry, so that it records the Picasso version and the step. Functions that select localizations rather than process them (e.g., ``postprocess.picked_locs``) return no metadata, and ``lib.fit_canvas`` updates the canvas of the metadata it is given in place. A loop that calls a step many times (e.g., once per pick) calls a private core function that skips the metadata. The Picasso windows keep and save the metadata these functions return rather than building their own. ``tests/test_locs_info_api.py`` checks the signatures of all public functions and the metadata of the steps.

Reporting Progress
^^^^^^^^^^^^^^^^^^

A public function that takes a while reports its progress through one argument, ``progress``, which is a tracker (an object with the ``picasso.lib.Progress`` interface, e.g., the dialog of a Picasso window), ``"console"`` (a tqdm bar for each step) or None (nothing is reported or printed). Annotate and document it in the same words, and say which steps it reports::

    def my_analysis(
        locs: pd.DataFrame,
        info: list[dict],
        progress: lib.Progress | Literal["console"] | None = None,
    ) -> pd.DataFrame:
        """...

        Parameters
        ----------
        progress : lib.Progress, "console" or None, optional
            Progress tracker, see ``picasso.lib.Progress``: e.g. a
            ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
            or None for no reporting. Reports "Picking localizations" over
            the picks, then "Fitting" over the groups. Default is None.
        """
        with lib.tracked_progress(progress, "My analysis") as progress:
            lib.start_phase(progress, "Picking localizations", n, unit="pick")
            for i in range(n):
                ...
                progress.set_value(i + 1)
            fit_something(..., progress=progress)  # starts its own steps

The rules:

- The function names and sizes each of its steps with ``lib.start_phase(progress, label, maximum, unit=...)``; callers never compute totals. A maximum of 0 means that the size is unknown (a busy indicator).
- ``progress.set_value(n)`` reports the number of items finished in the current step, never decreases within it and ends at the step's maximum.
- ``lib.tracked_progress`` turns ``"console"`` and None into a tracker and closes it afterwards; a tracker passed in by the caller is never closed, so pass it on to the Picasso functions you call, which add their own steps. Private helpers take the tracker as ``progress: lib.Progress``.
- Call the tracker only from the thread that called the function. Work done in other threads or processes is counted there and reported from the calling thread, e.g., while waiting for the futures.
- ``set_value`` may raise ``lib.OperationCanceled`` when the user presses *Cancel*: let it propagate and shut down pools and close files in ``finally``.
- Do not use tqdm or ``print`` directly: with None, nothing is printed.

In the Picasso windows, run the function in a task (``lib.run_task``) and pass it the task's ``progress``; a ``lib.ProgressDialog`` passed directly must be closed by the window when the function returns. For a worker thread with its own signals, subclass ``lib.MockProgress`` and emit them from ``set_value`` and ``zero_progress``.

``tests/test_progress_api.py`` checks the signature and the docstring of every public function. To test the steps of your function, pass the ``RecordingProgress`` of ``tests/_progress_tracking.py`` and check them with ``assert_phases_complete``; ``CancelingProgress`` simulates *Cancel*.

6. Submit a pull request through the GitHub website. See below for guidelines!

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated.
3. The pull request should work for all Python versions specified in ``pyproject.toml`` (currently 3.11–3.14).
4. Keep pull requests small and focused to make them easier to review. Make one pull request per issue/feature. If your pull request addresses multiple issues, consider splitting it into multiple pull requests.
5. Reference the related issue (if any was raised) in the pull request description (e.g. "Closes #123").
6. All CI checks must pass before the pull request can be merged. If your pull request fails CI checks, don't worry! The maintainers will work with you to resolve any issues. If you're not sure how to fix a CI failure, feel free to ask for help in the pull request comments.
7. Many thanks for contributing! We will review your pull request as soon as we can.
