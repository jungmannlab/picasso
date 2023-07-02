filetypes
=========

This section describes the different file format and name conventions used in Picasso.

Movie Files
-----------

Picasso accepts three types of raw movie files: TIFF (preferably from μManager), raw binary data (file extension “.raw”) and the Nikon format .nd2.

When loading raw binary files, the user will be prompted for movie metadata such as the number of frames, number of pixels, etc. Alternatively, this metadata can be supplied by an accompanying metadata file with the same filename as the raw binary file, but with the extension .yaml. See ``YAML Metadata Files`` for more details.

HDF5 Files
----------

HDF5 is a generic and efficient binary file format for storing data. In Picasso, HDF5 files are used for storing tabular data of localization properties with the file extension .hdf5. Furthermore, Picasso saves the statistical properties of groups of localizations in an HDF5 file.

Generally, several datasets can be stored within an HDF5 file. These datasets are accessible by specifying a path within the HDF5 file, similar to a path of an operating system. When saving localizations, Picasso stores tabular data under the path ``/locs``. When saving statistical properties of groups of localizations, Picasso saves the table under the path ``/groups``.


Importing HDF5 files in Pandas, MATLAB and Origin
-------------------------------------------------

In Pandas, use ``pandas.read_hdf(key='locs')`` or ``pandas.read_hdf(keys='group')``. 

In MATLAB, execute the command ``locs = h5read(filename, dataset)``. Replace dataset with ``/locs`` for localization files and with ``/groups`` for pick property files.

In Origin, select ``File > Import > HDF5`` or drag and drop the file into the main window.

Picasso uses ``h5py``. To load localizations, Picasso uses the function ``load_locs(filename)``` located in the ``io.py`` package of Picasso.

Localization HDF5 Files
-----------------------

Localization HDF5 files must always be accompanied by a YAML metadata file with the same filename, but with the extension .yaml. See ``YAML Metadata File`` for more details. The localization table is stored as a dataset of the HDF5 file in the path ``/locs``. This table can be visualized by opening the HDF5 file with ``Picasso: Filter``. The localization table can have an unlimited number of columns. Table 1 describes the meaning of Picasso’s main column names.

.. csv-table:: Table 1: Name, description and data type for the main columns used in Picasso.
   :file: table01.csv
   :widths: 20, 20, 20
   :header-rows: 1


HDF5 Pick Property Files
------------------------

When selecting ``File > Save pick properties`` in ``Picasso: Render``, the properties of picked regions are stored in an HDF5 file. Within the HDF5 file, the data table is stored in the path ``/groups``.
Each row in the “groups” table corresponds to one picked region. For each localization property (see Table 1), two columns are generated in the ``groups`` table: the mean and standard deviation of the respective column over the localizations in a pick region. For example, if the localization table contains a column ``len``, the “groups” table will contain a column ``len_mean`` and ``len_std``.
Furthermore, the following columns are included: ``group`` (the group identifier), ``n_events`` (the number of localizations in the region) and ``n_units`` (the number of units from a qPAINT measurement).

YAML Metadata Files
-------------------

YAML files are document-oriented text files that can be opened and changed with any text editor. In Picasso, YAML files are used to store metadata of movie or localization files.
Each localization HDF5 file must always be accompanied with a YAML file of the same filename, except for the extension, which is ``.yaml``. **Deleting this YAML metadata file will result in failure of the Picasso software!**

Raw binary files (i.e., with extension ``.raw``) may be accompanied by a YAML metadata file to store data about the movie dimensions, etc. While the metadata file, in this case, is not required, it reduces the effort of typing in this metadata each time the movie is loaded with ``Picasso: Localize``. To generate such a YAML metadata file, load the raw movie into ``Picasso: Localize``, then enter all required information in the appearing dialog. Check the checkbox ``Save info to yaml file`` and click ok. The movie will be loaded and the metadata saved in a YAML file. This file will be detected the next time this raw movie is loaded, and the metadata does not need to be entered again.