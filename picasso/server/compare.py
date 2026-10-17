import streamlit as st
from helper import fetch_db
import pandas as pd
import os
import numpy as np
from picasso import io
import plotly.express as px


@st.cache_data
def load_file(path: str):
    """Load a localization file and return it as a pandas DataFrame.

    Adds a column with the filename.

    Parameters
    ----------
    path : str
        Path to the localization file.

    Returns
    -------
    locs : pd.DataFrame
        The localizations, with an added ``file`` column.
    info : list of dicts
        Localizations metadata.
    """

    locs, info = io.load_locs(path)
    locs = pd.DataFrame(locs)
    locs["file"] = os.path.split(path)[-1]
    return locs, info


def get_file_family(file: str):
    """Return all files that belong to a parent file for picasso.

    E.g. for a folder with 'file.hdf5' and 'file_undrift.hdf5', when
    searching for 'file.hdf5', both files will be returned.

    Parameters
    ----------
    file : str
        Path to the parent file.

    Returns
    -------
    files : list of str
        Names of the ``.hdf5`` files sharing its base name.
    folder : str
        The folder they live in.
    """
    base = os.path.split(file)[1].split(".")[0]
    folder = os.path.dirname(file)

    files = os.listdir(folder)
    files = [f for f in files if f.startswith(base) and f.endswith(".hdf5")]

    return files, folder


def locs_per_frame_plot(hdf_dict: dict):
    """Plot the localizations per frame.

    Parameters
    ----------
    hdf_dict : dict
        Maps each file name to its localizations data frame.
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
        plot_df,
        x="frame",
        y="count",
        title="Locs per Frame",
        color="file",
        height=600,
    )

    fig.update_layout(
        legend=dict(
            x=0,
            y=-0.5,
        )
    )

    st.plotly_chart(fig)


def hist_plot(hdf_dict: dict, locs: pd.DataFrame):
    """Plot a histogram for a given hdf dictionary.

    Parameters
    ----------
    hdf_dict : dict
        Maps each file name to its localizations data frame.
    locs : pd.DataFrame
        Localizations, read for the list of selectable columns.
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
            plot_df,
            x=field,
            y="count",
            title=f"{field}",
            color="file",
            height=600,
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


def compare():  # noqa: C901
    """Compare streamlit page."""
    st.write("# Compare")

    st.write(
        "Compare multiple files from the database. All hdf files with the same"
        " base path as the movie will be selectable."
    )

    try:
        df = fetch_db()
        if df.empty:
            raise ValueError("no files in the database")

        files = df["filename"].unique()

        selected = st.multiselect(
            "Select files (Hover to see full path)",
            files,
        )

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
                        f"File **{file}** was not found. Please check that "
                        "this file exists."
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

    except ValueError:
        st.warning("Database empty. Process files first.")
