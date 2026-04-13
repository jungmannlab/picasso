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

If you are reporting a bug, please include:

* Your operating system name and version.
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

3. Follow the installation instructions for developers here: https://github.com/jungmannlab/picasso?tab=readme-ov-file#for-developers-local-editable-installation.

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

Committing & Pushing
^^^^^^^^^^^^^^^^^^^^^

4. Commit your changes using the imperative mood (e.g. "Fix bug" not "Fixed bug" or "Fixes bug"). Reference related issues in the commit message (#123 in the example below)::

    $ git add .
    $ git commit -m "Fix rendering crash for large datasets (#123)"
    $ git push origin fix/broken-rendering

5. Update the changelog (``changelog.md``) with a brief summary of your changes under the appropriate section.

6. Submit a pull request through the GitHub website. See below for guidelines!

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated.
3. The pull request should work for all Python versions specified in ``pyproject.toml`` (currently 3.10–3.14).
4. Keep pull requests small and focused to make them easier to review. Make one pull request per issue/feature. If your pull request addresses multiple issues, consider splitting it into multiple pull requests.
5. Reference the related issue (if any was raised) in the pull request description (e.g. "Closes #123").
6. All CI checks must pass before the pull request can be merged. If your pull request fails CI checks, don't worry! The maintainers will work with you to resolve any issues. If you're not sure how to fix a CI failure, feel free to ask for help in the pull request comments.
7. Many thanks for contributing! We will review your pull request as soon as we can.
