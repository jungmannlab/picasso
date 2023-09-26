import streamlit as st
from helper import _db_filename
from sqlalchemy import create_engine
import pandas as pd
import os
import numpy as np
from picasso import io
from picasso import render
from picasso import lib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_file(path: str):
    """Loads a localization file and returns as pandas Dataframe.
    Adds a column with the filename.

    Args:
        path (str): Path to localization file.
        file (str): filename
    """

    locs, info = io.load_locs(path)
    locs = pd.DataFrame(locs)
    locs["file"] = os.path.split(path)[-1]
    return locs, info


def get_file_family(file: str):
    """Returns all files that belong to a parent file for picasso.
    E.g. for a folder with 'file.hdf5' and 'file_undrift.hdf5',
    when searching for 'file.hdf5', both files will be returned.

    Args:
        file (str): Path to file.
    """
    base = os.path.split(file)[1].split(".")[0]
    folder = os.path.dirname(file)

    files = os.listdir(folder)
    files = [f for f in files if f.startswith(base) and f.endswith(".hdf5")]

    return files, folder


def locs_per_frame_plot(hdf_dict: dict):
    """Plots the localizations per frame.

    Args:
        hdf_dict (dict): Dictionary with hdf summary information
    """
    smooth = st.number_input("Smooth", value=100, min_value=1, max_value=1000)
    summary = []

    for f, df in hdf_dict.items():
        d = df[["frame", "photons"]].groupby("frame").count()
        d.columns = ["count"]
        d = d.rolling(smooth).mean()
        d["file"] = f
        d = d.reset_index()

        summary.append(d)

    plot_df = pd.concat(summary, axis=0)

    fig = px.line(
        plot_df, x="frame", y="count", title="Locs per Frame", color="file", height=600
    )

    fig.update_layout(
        legend=dict(
            x=0,
            y=-0.5,
        )
    )

    st.plotly_chart(fig)


def hist_plot(hdf_dict: dict, locs: pd.DataFrame):
    """Plots a histogram for a given hdf dictionary.

    Args:
        hdf_dict (dict): Dictionary with summary stats per file.
        locs (pd.DataFrame): pandas Dataframe with localizations.
    """

    c1, c2, c3, c4 = st.columns(4)

    fields = locs.columns
    field = c1.selectbox("Select field", fields)

    try:
        n_bins = c2.number_input(
            "Number of bins", value=100, min_value=1, max_value=200
        )

        all_f = {}
        for f, df in hdf_dict.items():
            all_f[f] = df[field].values

        all_values = np.concatenate(list(all_f.values()))

        min_ = float(np.min(all_values))
        max_ = float(np.max(all_values))

        upper = float(np.percentile(all_values, 99))
        lower = float(np.percentile(all_values, 1))

        min_value = c3.number_input(
            "Min value", value=lower, min_value=min_, max_value=max_
        )
        max_value = c4.number_input(
            "Max value", value=upper, min_value=min_value, max_value=max_
        )

        bins = np.linspace(min_value, max_value, n_bins)

        summary = []
        for f, vals in all_f.items():

            counts, _ = np.histogram(vals, bins=bins)
            sub_df = pd.DataFrame([bins[1:] + bins[0] / 2, counts]).T

            sub_df.columns = [field, "count"]
            sub_df["file"] = f
            summary.append(sub_df)

        plot_df = pd.concat(summary, axis=0)

        fig = px.line(
            plot_df, x=field, y="count", title=f"{field}", color="file", height=600
        )

        fig.update_layout(
            legend=dict(
                x=0,
                y=-0.5,
            )
        )

        st.plotly_chart(fig)
    except Exception as e:
        st.warning(f"An error occured plotting field **{field}**.\n {e}")


def compare():
    """Compare streamlit page."""
    st.write("# Compare")

    st.write(
        "Compare multiple files from the database. All hdf files with the same base path as the movie will be selectable."
    )

    engine = create_engine("sqlite:///" + _db_filename(), echo=False)

    try:
        df = pd.read_sql_table("files", con=engine)

        files = df["filename"].unique()

        selected = st.multiselect("Select files (Hover to see full path)", files)

        if len(selected) > 0:

            file_dict = {}
            hdf_dict = {}
            for file in selected:
                try:
                    c1, f1 = get_file_family(file)
                    file_dict[file] = st.multiselect(
                        f"Select hdf file for {file}",
                        c1,
                        None,
                    )

                    if file_dict[file] is not None:
                        for _ in file_dict[file]:
                            path = os.path.dirname(file)

                            locs_filename = os.path.join(path, _)

                            with st.spinner("Loading files"):
                                locs, info = load_file(locs_filename)
                                hdf_dict[locs_filename] = locs
                except FileNotFoundError:
                    st.error(
                        f"File **{file}** was not found. Please check that this file exists."
                    )

            st.write("## Plot")

            if len(hdf_dict) > 0:
                with st.spinner("Generating plots"):
                    plot = st.selectbox(
                        "Select plot", ["Localizations per frame", "Histogram"]
                    )
                    if plot == "Localizations per frame":
                        locs_per_frame_plot(hdf_dict)
                    else:
                        hist_plot(hdf_dict, locs)
            else:
                st.warning("Please select HDF files.")

    except ValueError as e:

        st.warning("Database empty. Process files first.")
