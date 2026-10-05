import streamlit as st
import os
import pandas as pd
from multiprocessing import Process
import time
from sqlalchemy import create_engine
from picasso import localize
from helper import fetch_watcher, fetch_db
import psutil
import subprocess
from datetime import datetime
from picasso.__main__ import _localize

DEFAULT_UPDATE_TIME = 1

FILETYPES = (".raw", ".ome.tif", ".ims")


class aclass:
    """Namespace object whose attributes are the keyword arguments given.

    Used to hand the localize settings to code that expects an argparse
    namespace.
    """

    def __init__(self, **entries):
        """Set one attribute per keyword argument.

        Parameters
        ----------
        **entries
            The attributes to set.
        """
        self.__dict__.update(entries)


def check_new(path: str, processed: dict, logfile: str):
    """Check which files in a folder are not processed yet.

    Files are considered processed if they have a ``_locs.hdf5`` file.

    Parameters
    ----------
    path : str
        Folder to check.
    processed : dict
        Files that are already processed, keyed by normalized path.
    logfile : str
        Path to the logfile new findings are reported to.

    Returns
    -------
    new : list of str
        Movie files that still need processing.
    processed : dict
        ``processed``, with the files that turned out to be done already
        added to it.
    """

    all_ = os.listdir(path)
    all_ = [os.path.join(path, _) for _ in all_]

    new = [
        _
        for _ in all_
        if (
            os.path.normpath(_) not in processed.keys()
            and _.endswith(FILETYPES)
        )
    ]
    locs = [_ for _ in all_ if _.endswith("_locs.hdf5")]

    print_to_file(
        logfile,
        (
            f"{datetime.now()} Checking: {len(all_)} files, {len(new)} "
            f"unprocessed with valid ending and {len(locs)} `_locs.hdf5` "
            " files in {path}.",
        ),
    )

    for _ in new:
        base, ext = os.path.splitext(_)
        for ref in locs:
            if base.startswith(ref):
                processed[_] = True
                new.remove(_)
                break

    return new, processed


def wait_for_change(file: str):
    """Block until a file stops growing.

    Parameters
    ----------
    file : str
        Path to the file being written.
    """
    print(f"Waiting for {file}")
    filesize = os.path.getsize(file)
    writing = True
    while writing:
        time.sleep(2)
        new_filesize = os.path.getsize(file)
        if filesize == new_filesize:
            writing = False
        else:
            filesize = new_filesize

    print(f"File {file} complete.")


def get_children_files(file: str, checked: list):
    """List the files sitting next to ``file`` in its folder.

    Parameters
    ----------
    file : str
        Path whose folder is listed.
    checked : list
        Files that were already checked; accepted for the caller's
        bookkeeping.

    Returns
    -------
    files_in_folder : list of str
        Absolute paths of every file in that folder.
    """
    dir_ = os.path.dirname(file)
    files_in_folder = [
        os.path.abspath(os.path.join(dir_, _)) for _ in os.listdir(dir_)
    ]
    # Multiple ome tifs; Pos0.ome.tif', Pos0_1.ome.tif', Pos0_2.ome.tif'
    files_in_folder = [
        _
        for _ in files_in_folder
        if (
            _.startswith(file[:-8])
            and _ not in checked
            and _.endswith(".ome.tif")
            and "MMStack_Pos0" in _
        )
    ]

    for _ in files_in_folder:
        wait_for_change(_)

    return files_in_folder


def wait_for_completion(file: str):
    """Wait until a file - and, for OME-TIFFs, its continuation files - is
    completely written.

    Parameters
    ----------
    file : str
        Path to the movie being acquired.

    Returns
    -------
    checked : list of str
        The file and the continuation files waited for; empty for a
        single-file movie.
    """

    wait_for_change(file)

    if file.endswith(".ome.tif"):
        checked = [file]

        time.sleep(2)

        children = get_children_files(file, checked)
        checked.extend(children)

        while len(children) > 0:
            children = get_children_files(file, checked)
            checked.extend(children)

    else:
        checked = []

    return checked


def print_to_file(path, text):
    """Append a line to the watcher's logfile.

    Parameters
    ----------
    path : str
        Path to the logfile.
    text : str
        The line to append.
    """
    with open(path, "a") as f:
        f.write(text)
        f.write("\n")


def check_new_and_process(  # noqa: C901
    settings_list: dict,
    path: str,
    command: str,
    logfile: str,
    existing: list,
    update_time: int,
):
    """Watch a folder for new movies and localize them as they appear.

    Runs until interrupted.

    Parameters
    ----------
    settings_list : dict
        Localize settings, one entry per processing step.
    path : str
        Folder to watch.
    command : str
        Command to execute after processing a file.
    logfile : str
        Path to the logfile.
    existing : list
        Files already present that should not be processed.
    update_time : int
        How long to wait between scans of the folder, in minutes.
    """

    print_to_file(logfile, f"{datetime.now()} Started watcher for {path}.")
    print_to_file(logfile, f"{datetime.now()} Settings {settings_list}.")

    processed = {}

    for _ in existing:
        processed[_] = True

    while True:
        new, processed = check_new(path, processed, logfile)

        if len(new) > 0:
            file = os.path.abspath(new[0])
            print_to_file(logfile, f"{datetime.now()} New file {file}")
            children = wait_for_completion(file)
            print_to_file(logfile, f"{datetime.now()} Children {children}")
            try:
                for settings in settings_list:

                    if len(settings_list) > 1:
                        print_to_file(
                            logfile,
                            (
                                f"{datetime.now()} Processing group "
                                f"{settings['suffix']}",
                            ),
                        )

                    settings["files"] = file

                    args_ = aclass(**settings)
                    _localize(args_)

                if command != "":

                    if "$FILENAME" in command:
                        to_execute = command[:]
                        to_execute = to_execute.replace(
                            "$FILENAME",
                            f'"{file}"',
                        )

                    print_to_file(
                        logfile,
                        f"{datetime.now()} Executing {to_execute}.",
                    )

                    subprocess.run(to_execute)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print_to_file(
                    logfile,
                    f"{datetime.now()} Exception {e} occured.",
                )

            processed[file] = True

            for _ in children:
                processed[_] = True

            print_to_file(logfile, f"{datetime.now()} File {file} processed.")
            print(f"{datetime.now()} File {file} processed.")

        time.sleep(update_time * 60)


def watcher():  # noqa: C901
    """
    Streamlit page to show the watcher page.
    """
    st.write("# Watcher")
    st.text(
        "- Set up a file watcher to process files in a folder with pre-defined"
        " settings automatically."
    )
    st.text(
        "- All new files and raw files that aren't yet in the database will be"
        " processed."
    )
    st.text(
        "- You can define different parameter groups so that a file will be"
        " processed with different settings."
    )
    st.text("- You can also chain custom commands to the watcher.")
    st.text(
        f"- The watcher will check for the following filetypes: {FILETYPES}"
    )

    st.write("## Existing watchers")
    df_ = fetch_watcher()
    df = df_.copy()
    if len(df) > 0:
        df["running"] = [psutil.pid_exists(_) for _ in df["process id"]]
        st.dataframe(df)
        if df["running"].sum() != len(df):
            if st.button(
                "Remove non-running watchers", icon=":material/delete_sweep:"
            ):
                df = df[df["running"]]
                engine = create_engine(
                    "sqlite:///" + localize.db_filename(), echo=False
                )
                df.to_sql(
                    "watcher",
                    con=engine,
                    if_exists="replace",
                    index=False,
                )

                st.success("Removed. Please refresh this page.")

                st.stop()

        logfile = st.selectbox("Select logfile", df["logfile"].iloc[::-1])
        with st.expander("Log"):
            with open(logfile, "r") as f:
                text = f.readlines()
                st.text("".join(text))

    else:
        st.write("None")

    st.write("## New watcher")
    folder = st.text_input("Enter folder to watch.", os.getcwd())

    if not os.path.isdir(folder):
        st.error("Not a valid path.")
    else:
        with st.expander("Settings"):

            n_columns = int(
                st.number_input(
                    "Number of Parameter Groups",
                    min_value=1,
                    max_value=10,
                    step=1,
                )
            )

            if n_columns > 1:
                st.text(
                    "Parameter groups will be indiciated with a `pg` in the "
                    "filename, e.g. `filename_pg_1_locs.hdf`"
                )

            columns = st.columns(n_columns)
            settings = {}

            for i in range(n_columns):
                col = columns[i]
                settings[i] = {}
                if n_columns > 1:
                    col.write(f"Group {i+1} (pg_{i+1})")

                settings[i]["box"] = col.number_input(
                    label="Box side length:", value=7, key=f"box_{i}"
                )
                settings[i]["min_net_gradient"] = col.number_input(
                    label="Min. Net Gradient:",
                    value=5000,
                    min_value=0,
                    max_value=10000,
                    key=f"gradient_{i}",
                )

                settings[i]["em_gain"] = col.number_input(
                    label="EM Gain:",
                    value=1,
                    min_value=1,
                    max_value=1000,
                    key=f"em_{i}",
                )

                settings[i]["baseline"] = col.number_input(
                    label="Baseline:", value=100, key=f"baseline_{i}"
                )
                settings[i]["sensitivity"] = col.number_input(
                    label="Sensitivity:", value=1.00, key=f"sensitivity_{i}"
                )
                settings[i]["qe"] = col.number_input(
                    label="Quantum Efficiency:",
                    value=0.9,
                    min_value=0.01,
                    max_value=1.0,
                    key=f"qe_{i}",
                )
                settings[i]["pixelsize"] = col.number_input(
                    label="Pixelsize (nm):",
                    value=130,
                    min_value=10,
                    max_value=500,
                    key=f"pixelsize_{i}",
                )

                settings[i]["methods"] = col.selectbox(
                    "Method", options=["lq", "mle"], key=f"method_{i}"
                )

                settings[i]["undrift"] = col.number_input(
                    "Segmentation size for undrift",
                    value=1000,
                    min_value=100,
                    max_value=10000,
                    key=f"segmentation_{i}",
                )

                if col.checkbox("3D", key=f"3d_checkbox_{i}"):
                    col.info("3D will set fit method to lq-3d.")

                    settings[i]["methods"] = "lq-3d"

                    settings[i]["calib_file"] = col.text_input(
                        "Enter path to calibration file.", os.getcwd()
                    )

                    if not os.path.isfile(settings[i]["calib_file"]):
                        col.error("Not a valid file.")

                    settings[i]["magnification_factor"] = col.number_input(
                        label="Magnification factor:",
                        value=0.79,
                        key=f"magfac_{i}",
                    )
                else:
                    settings[i]["calib_file"] = None
                    settings[i]["magnification_factor"] = None

            if st.checkbox("Chain Custom command"):
                st.text("Allows to execute a custom command via shell.")
                st.text("You can pass the filename with $FILENAME.")
                command = st.text_input("Command", "")
            else:
                command = ""

            update_time = st.number_input(
                "Update time (scan every x-th minute):", DEFAULT_UPDATE_TIME
            )

            logfile_dir = os.path.dirname(localize.db_filename())
            now_str = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
            logfile = os.path.join(logfile_dir, f"{now_str}_watcher.log")
            logfile = st.text_input("Logfile", logfile)

            if st.button("Submit", icon=":material/send:"):

                settings_list = []

                for i in range(n_columns):
                    settings_selected = {}
                    settings_selected["box_side_length"] = settings[i]["box"]
                    settings_selected["gradient"] = settings[i][
                        "min_net_gradient"
                    ]
                    settings_selected["gain"] = settings[i]["em_gain"]
                    settings_selected["baseline"] = settings[i]["baseline"]
                    settings_selected["sensitivity"] = settings[i][
                        "sensitivity"
                    ]
                    settings_selected["qe"] = settings[i]["qe"]
                    settings_selected["pixelsize"] = settings[i]["pixelsize"]
                    settings_selected["fit_method"] = settings[i]["methods"]
                    settings_selected["drift"] = settings[i]["undrift"]
                    if settings[i]["calib_file"]:
                        settings_selected["zc"] = settings[i]["calib_file"]
                    if settings[i]["magnification_factor"]:
                        settings_selected["mf"] = settings[i][
                            "magnification_factor"
                        ]

                    settings_selected["database"] = True
                    if n_columns == 1:
                        suffix = ""
                    else:
                        suffix = f"_pg_{i+1}"
                    settings_selected["suffix"] = suffix
                    settings_list.append(settings_selected)

                st.write(settings_list)

                existing = fetch_db()

                if len(existing) > 0:
                    existing = existing["filename"].tolist()
                else:
                    existing = []

                p = Process(
                    target=check_new_and_process,
                    args=(
                        settings_list,
                        folder,
                        command,
                        logfile,
                        existing,
                        update_time,
                    ),
                )
                p.start()

                st.success("Started watcher.")
                display = st.empty()

                df = pd.DataFrame(
                    {
                        "process id": [p.pid],
                        "started": [datetime.now()],
                        "folder": [folder],
                        "logfile": [logfile],
                    }
                )

                engine = create_engine(
                    "sqlite:///" + localize.db_filename(), echo=False
                )
                df.to_sql(
                    "watcher",
                    con=engine,
                    if_exists="append",
                    index=False,
                )

                for reset in range(3):
                    display.success(f"Restarting in {3-reset}.")
                    time.sleep(1)
                display.success("Please refresh page.")
                st.stop()
