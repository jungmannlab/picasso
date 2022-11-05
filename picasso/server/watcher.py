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
import sys

DEFAULT_UPDATE_TIME = 1
DEFAULT_RUNNING_TIME = 12

FILETYPES = (".raw", ".tif", ".ims")
from picasso.__main__ import _localize
from picasso.io import load_movie


class aclass:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def check_new(path: str, processed: dict, logfile: str):
    """Check if files in a folder and its direct subfolders are not 
    processed yet.
    Subfolders are checked for compatibilty with MicroManager.
    Files are considered processed if they have a _locs.hdf5 file.
    Paths to detected files are ordered alphabetically to allow for finding
    of the children files later.

    Args:
        path (str): Folder to check.
        processed (dict): Dict of files that are already processed.
        logfile (str): Path to logfile.

    Returns:
        _type_: _description_
    """

    # files directly in path
    all_ = os.listdir(path)
    all_ = [os.path.join(path, _) for _ in all_]

    # list of paths to the direct subfolders in path
    subfolders = [_ for _ in all_ if os.path.isdir(_)]

    all_sub = []
    for folder in subfolders:
        files = os.listdir(folder)
        all_sub += [os.path.join(folder, _) for _ in files]

    all_ = all_ + all_sub

    new = [
        _
        for _ in all_
        if os.path.normpath(_)
        not in processed.keys()
        and _.endswith(FILETYPES)
    ]
    locs = [_ for _ in all_ if _.endswith("_locs.hdf5")]

    print_to_file(
        logfile,
        (
            f"{datetime.now()} Checking: {len(all_)} files,"
            f"  {len(new)} unprocessed with valid ending and "
            f"{len(locs)} `_locs.hdf5` files in {path}."
        ),
    )

    for _ in new:
        base, ext = os.path.splitext(_)
        for ref in locs:
            if base.startswith(ref):
                processed[_] = True
                new.remove(_)
                break

    new = sorted(new)
    return new, processed


def wait_for_change(file: str):
    """Helper function that checks if a file is changing the size.

    Args:
        file (str): Path to file.
    """
    print(f"Waiting for {file}")
    filesize = os.path.getsize(file)
    writing = True
    while writing:
        time.sleep(15)
        new_filesize = os.path.getsize(file)
        if filesize == new_filesize:
            writing = False
        else:
            filesize = new_filesize

    print(f"File {file} complete.")


def get_children_files(file: str, checked: list):
    """Helper function that extracts files with the same start and same ending.

    Args:
        file (str): Path to check.
        checked (list): List with files that are already checked.
    """
    dir_ = os.path.dirname(file)
    files_in_folder = [
        os.path.abspath(os.path.join(dir_, _))
        for _ in os.listdir(dir_)
    ]
    # Multiple ome tifs; Pos0.ome.tif', Pos0_1.ome.tif', Pos0_2.ome.tif'
    if file.endswith(".ome.tif"):
        files_in_folder = [
            _
            for _ in files_in_folder
            if _.startswith(file[:-8])
            and _ not in checked
            and _.endswith(".ome.tif")
            and 'MMStack_Pos0' in _
        ]
    # Multiple NDTiffStack files
    elif file.endswith(".tif"):
        files_in_folder = [
            _
            for _ in files_in_folder
            if _.startswith(file[:-4])
            and _ not in checked
            and _.endswith(".tif")
            and 'NDTiffStack' in _
        ]

    for _ in files_in_folder:
        wait_for_change(_)

    return files_in_folder


def wait_for_completion(file: str):
    """Helper function that waits until a file is completely written.

    Args:
        file (str): Filepath.
    """

    wait_for_change(file)

    if file.endswith(".ome.tif"):
        checked = [file]

        time.sleep(15)

        children = get_children_files(file, checked)
        checked.extend(children)

        while len(children) > 0:
            children = get_children_files(file, checked)
            checked.extend(children)

    else:
        checked = []

    return checked


def print_to_file(path, text):
    with open(path, "a") as f:
        f.write(text)
        f.write("\n")


def check_new_and_process(
    settings_list: dict,
    path: str,
    command: str,
    logfile: str,
    existing: list,
    update_time: int,
    running_time: int,
):
    """
    Checks a folder for new files and processes them with defined settigns.
    Args:
        settings_list (list): List of dictionaries with settings.
        path (str): Path to folder.
        command (str): Command to execute after processing.
        logfile (str): Path to logfile.
        existing (list): existing files
        update_time (int): Refresh every x minutes
        running_time (int): Total running time, watcher turns off after that
    """

    t0 = time.time()
    running_time *= 3600 # convert to seconds

    print_to_file(logfile, f"{datetime.now()} Started watcher for {path}.")
    print_to_file(logfile, f"{datetime.now()} Settings {settings_list}.")

    processed = {}

    for _ in existing:
        processed[_] = True

    while dt < running_time:
        new, processed = check_new(path, processed, logfile)

        if len(new) > 0:
            file = os.path.abspath(new[0])
            print_to_file(logfile, f"{datetime.now()} New file {file}")
            children = wait_for_completion(file)
            print_to_file(logfile, f"{datetime.now()} Children {children}")
            try:
                # do not localize tif files with only a single frame
                n_frames = len(load_movie(file)[0])
                if n_frames > 1:
                    for settings in settings_list:
                        if len(settings_list) > 1:
                            print_to_file(
                                logfile,
                                (
                                    f"{datetime.now()} Processing group "
                                    f" {settings['suffix']}"
                                ),
                            )

                        settings["files"] = file

                        args_ = aclass(**settings)
                        _localize(args_)

                    if command != "":

                        if "$FILENAME" in command:
                            to_execute = command[:]
                            to_execute = to_execute.replace(
                                "$FILENAME", f'"{file}"'
                            )

                        print_to_file(
                            logfile,
                            f"{datetime.now()} Executing {to_execute}."
                        )

                        subprocess.run(to_execute)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print_to_file(
                    logfile, f"{datetime.now()} Exception {e} occured."
                )

            processed[file] = True

            for _ in children:
                processed[_] = True

            print_to_file(logfile, f"{datetime.now()} File {file} processed.")
            print(f"{datetime.now()} File {file} processed.")

        time.sleep(update_time * 60)
        dt = time.time() - t0


def watcher():
    """
    Streamlit page to show the watcher page.
    """
    st.write("# Watcher")
    st.write(
        " \n- Set up a file watcher to process files in a folder with pre-defined"
        "  settings automatically."
        " \n- All new files and raw files that aren't yet in the database will be"
        "  processed."
        " \n- You can define different parameter groups so that a file will be"
        "  processed with different settings."
        " \n- You can also chain custom commands to the watcher."
        f" \n- The watcher will check for the following filetypes: {FILETYPES}"
    )

    st.write("## Existing watchers")
    df_ = fetch_watcher()
    df = df_.copy()
    if len(df) > 0:
        df["running"] = [psutil.pid_exists(_) for _ in df["process id"]]
        st.dataframe(df)
        if df["running"].sum() != len(df):
            if st.button("Remove non-running watchers."):
                df = df[df["running"]]
                engine = create_engine(
                    "sqlite:///" + localize._db_filename(), echo=False
                )
                df.to_sql(
                    "watcher", con=engine, if_exists="replace", index=False
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
    folder_exists = False

    if not os.path.isdir(folder):
        folder_dir = os.path.dirname(folder)
        if os.path.isdir(folder_dir):
            # make the new folder
            if st.button("Create a new folder?"):
                os.mkdir(folder)
                st.write("The folder was created.")
                folder_exists = True
        else:
            st.error("Directory not found, new folder could not be created.")
    else:
        folder_exists = True

    if folder_exists:
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
                    "Parameter groups will be indiciated with a `pg` in the"
                    "  filename, e.g. `filename_pg_1_locs.hdf`"
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
                    label="Sensitivity:", value=1.0, key=f"sensitivity_{i}"
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
            running_time = st.number_input(
                "Total running time (hours):",
                value=DEFAULT_RUNNING_TIME,
                min_value=1,
                max_value=72,
            )

            logfile_dir = os.path.dirname(localize._db_filename())
            now_str = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
            logfile = os.path.join(logfile_dir, f"{now_str}_watcher.log")
            logfile = st.text_input("Logfile", logfile)

            if st.button("Submit"):

                settings_list = []

                for i in range(n_columns):
                    settings_selected = {}
                    settings_selected["box_side_length"] = settings[i]["box"]
                    settings_selected["gradient"] = (
                        settings[i]["min_net_gradient"]
                    )
                    settings_selected["gain"] = settings[i]["em_gain"]
                    settings_selected["baseline"] = settings[i]["baseline"]
                    settings_selected["sensitivity"] = (
                        settings[i]["sensitivity"]
                    )
                    settings_selected["qe"] = settings[i]["qe"]
                    settings_selected["pixelsize"] = settings[i]["pixelsize"]
                    settings_selected["fit_method"] = settings[i]["methods"]
                    settings_selected["drift"] = settings[i]["undrift"]
                    if settings[i]["calib_file"]:
                        settings_selected["zc"] = settings[i]["calib_file"]
                    if settings[i]["magnification_factor"]:
                        settings_selected["mf"] = (
                            settings[i]["magnification_factor"]
                        )

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
                    existing = existing['filename'].tolist()
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
                        running_time,
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
                    "sqlite:///" + localize._db_filename(), echo=False
                )
                df.to_sql(
                    "watcher", con=engine, if_exists="append", index=False
                )

                for reset in range(3):
                    display.success(f"Restarting in {3-reset}.")
                    time.sleep(1)
                display.success("Please refresh page.")
                st.stop()
