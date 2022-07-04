import streamlit as st
from helper import fetch_db
import os
import pandas as pd
import datetime
from multiprocessing import Process
import time
from collections import namedtuple
from sqlalchemy import create_engine
from picasso import localize
from helper import fetch_watcher
import psutil
import subprocess

UPDATE_TIME = 60

FILETYPES = (".raw", ".ome.tif", ".ims")
from picasso.__main__ import _localize


class aclass:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def check_new(path: str, processed: list):
    """Check if files in a folder are not processed yet.
    Files are considered processed if they have a _locs.hdf5 file.

    Args:
        path (str): Folder to check.
        processed (list): List of files that are already processed.

    Returns:
        _type_: _description_
    """

    all_ = os.listdir(path)

    new = [_ for _ in all_ if _ not in processed and _.endswith(FILETYPES)]

    for _ in new:
        base, ext = os.path.splitext(_)

        if base + "_locs.hdf5" in all_:
            processed[_] = True
            new.remove(_)

    return new


def wait_for_change(file: str):
    """Helper function that checks if a file is changing the size.

    Args:
        file (str): Path to file.
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
            
    print(f'File {file} complete.')


def get_children_files(file: str, checked: list):
    """Helper function that extracts files with the same start and same ending.

    Args:
        file (str): Path to check.
        checked (list): List with files that are already checked.
    """
    dir_ = os.path.dirname(file)
    files_in_folder = [os.path.abspath(os.path.join(dir_, _)) for _ in os.listdir(dir_)]
    # Multiple ome tifs; Pos0.ome.tif', Pos0_1.ome.tif', Pos0_2.ome.tif'
    files_in_folder = [
        _
        for _ in files_in_folder
        if _.startswith(file[:-8]) and _ not in checked and _.endswith(".ome.tif")
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

        time.sleep(2)
        children = get_children_files(file, checked)
        checked.extend(children)

        while len(children) > 0:
            children = get_children_files(file, checked)
            checked.extend(children)

    else:
        checked = []

    return checked


def check_new_and_process(settings: dict, path: str):
    """
    Checks a folder for new files and processes them with defined settigns.
    Args:
        settings (dict): Dictionary with settings.
        path (str): Path to folder.
    """
    print(f"Started watcher for {path}")
    print(f"Settings {settings}")
    processed = {}

    while True:
        new = check_new(path, processed)

        if len(new) > 0:
            file = os.path.abspath(os.path.join(path, new[0]))
            children = wait_for_completion(file)

            settings["files"] = file

            args_ = aclass(**settings)
            _localize(args_)

            cmd = settings["command"]

            if "$FILENAME" in cmd:
                cmd = cmd.replace("$FILENAME", f'"{file}"')

            if cmd != "":
                print(f"Executing {cmd}")
                subprocess.run(cmd)

            processed[file] = True

            for _ in children:
                processed[_] = True

        time.sleep(UPDATE_TIME)


def watcher():
    """
    Streamlit page to show the watcher page.
    """
    st.write("# Watcher")
    st.write(
        "Set up a file watcher to process files in a folder with pre-defined settings automatically."
    )
    st.write(
        "All raw files and new files that haven't been processed will be processed."
    )
    st.write("Use different folders to process files with different settings.")
    st.write("You can also chain custom commands to the watcher.")

    st.write("## Existing watchers")
    df_ = fetch_watcher()
    df = df_.copy()
    if len(df) > 0:
        df["running"] = [psutil.pid_exists(_) for _ in df["process id"]]
        st.table(df)
        if df["running"].sum() != len(df):
            if st.button("Remove non-running watchers."):
                df = df[df["running"]]
                if len(df) == 0:
                    df = pd.DataFrame({"process id": [], "started": [], "folder": []})
                engine = create_engine(
                    "sqlite:///" + localize._db_filename(), echo=False
                )
                df.to_sql("watcher", con=engine, if_exists="replace", index=False)

                st.stop()
    else:
        st.write("None")

    st.write("## New watcher")
    folder = st.text_input("Enter folder to watch.", os.getcwd())

    if not os.path.isdir(folder):
        st.error("Not a valid path.")
    else:
        with st.expander("Settings"):
            box = st.number_input(label="Box side length:", value=7)
            min_net_gradient = st.number_input(
                label="Min. Net Gradient:", value=5000, min_value=0, max_value=10000
            )

            em_gain = st.number_input(
                label="EM Gain:", value=1, min_value=1, max_value=1000
            )
            baseline = st.number_input(label="Baseline:", value=100)
            sensitivity = st.number_input(label="Sensitivity:", value=1)
            qe = st.number_input(
                label="Quantum Efficiency:", value=0.9, min_value=0.01, max_value=1.0
            )
            pixelsize = st.number_input(
                label="Pixelsize (nm):", value=130, min_value=10, max_value=500
            )

            methods = st.selectbox("Method", options=["lq", "mle"])

            undrift = st.number_input(
                "Segmentation size for undrift",
                value=1000,
                min_value=100,
                max_value=10000,
            )

            if st.checkbox("3D"):

                calib_file = st.text_input(
                    "Enter path to calibration file.", os.getcwd()
                )

                if not os.path.isfile(calib_file):
                    st.error("Not a valid file.")

                magnification_factor = st.slider(
                    label="Magnification factor:", value=0.79
                )
            else:
                calib_file = None
                magnification_factor = None

            if st.checkbox("Custom command"):
                st.text("Allows to execute a custom command via shell.")
                st.text("You can pass the filename with $FILENAME.")
                command = st.text_input("Command", "")
            else:
                command = ""

        if st.button("Submit"):
            settings = {}
            settings["box_side_length"] = box
            settings["gradient"] = min_net_gradient
            settings["gain"] = em_gain
            settings["baseline"] = baseline
            settings["sensitivity"] = sensitivity
            settings["qe"] = qe
            settings["pixelsize"] = pixelsize
            settings["fit_method"] = methods
            settings["drift"] = undrift
            if calib_file:
                settings["zpath"] = calib_file
            if magnification_factor:
                settings["magnification_factor"] = magnification_factor

            settings["command"] = command

            st.write(settings)

            settings["database"] = True

            p = Process(
                target=check_new_and_process,
                args=(settings, folder),
            )
            p.start()

            st.success("Started watcher.")
            display = st.empty()

            df = pd.DataFrame(
                {
                    "process id": [p.pid],
                    "started": [datetime.datetime.now()],
                    "folder": [folder],
                }
            )

            engine = create_engine("sqlite:///" + localize._db_filename(), echo=False)
            df.to_sql("watcher", con=engine, if_exists="append", index=False)

            for reset in range(3):
                display.success(f"Restarting in {3-reset}.")
                time.sleep(1)
            display.success("Please refresh page.")
            st.stop()
