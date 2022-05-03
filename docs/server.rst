server
======

.. image:: ../docs/server_main.png
   :scale: 10 %
   :alt: Main


Functionality
-------------
Picasso Server allows to continuously track performance metrics of your super-resolution experiments.
It does so by recording the metadata, derived summary statistics, and results of post-processing algorithms (such as the NeNA value) to a local SQL database.

You can also set up a ``FileWatcher`` that allows to continuously process new files in a folder.


Database
--------
The local SQL database will be stored in the ``.alphapept`` folder in your home directory. To directly access the database
the tool `DB Browser for SQLite <https://sqlitebrowser.org>`_ is recommended.

Localize
--------
The integration within ``Localize`` is by activating the ``Add to Database``- checkbox in the ``Parameters``.

.. image:: ../docs/server_localize.png
   :scale: 20 %
   :alt: Localize

Server
------
Server is a tool to interactively explore the database and to enable continuous processing workflows.
When launching server, the command line will start and show you the local IP address on where the server is running.
You can use this IP to e.g., connect to ``Picasso Server`` from within a network.

.. image:: ../docs/server_cmd.png
   :scale: 40 %
   :alt: CMD

When launching picasso server, it should open your default browser automatically and redirect to the Picasso Server page.
In case you close the website tab or the browser, ``Picasso Server`` will run in the background until the command line window is closed.
You can go back to the website by entering the link.

Status
~~~~~~
Displays the current database status and documentation.
This page also allows to manually add already processed files to the database.

History
~~~~~~~

.. image:: ../docs/server_history.png
   :scale: 10 %
   :alt: History

Explore summary statistics of processed files.
It is possible to filter by filename and group.
The following modes of display exist:

- Table: A table of the results. Each field contains a barplot showing the value relative to the column's maximum.
- Scatter: Scatterplot of results. This also allows to draw trendlines
- Boxplot: Daily Boxplot

Compare
~~~~~~~
Compare two files against each other.

**The database will store the file path when it is localized. If the file is moved, it will not be selectable.**

.. image:: ../docs/server_compare.png
   :scale: 10 %
   :alt: Localize

To compare experiments, select one or multiple experiments from the dropdown menu.
If multiple hdf files are present, you can select hdf files that belong to the same file family.

**Comparing files will load the entire hdf file and could mean that one is comparing millions of localizations.
Creating the plots might, therefore, not be instantaneous. **

- Localizations per frame: this will plot the localization per frame. This is useful to inspect the performance of an experiment over time.
- Histogram: creates a histogram for the population. This is useful e.g., for comparing background signals.

.. image:: ../docs/server_compare_histogram.png
   :scale: 10 %
   :alt: Histogram

Watcher
~~~~~~~
Set up a watcher to automatically process files in a folder with pre-defined settings.

* All raw files in the folder that haven't been processed will be processed.
* Use different folders to process files with different settings.

Preview
~~~~~~~
Preview will render the super-resolution data in the browser.

.. image:: ../docs/server_preview.png
   :scale: 10 %
   :alt: Histogram

**The database will store the file path when it is localized. If the file is moved, it will not be selectable.**
