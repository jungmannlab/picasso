import streamlit as st
from helper import fetch_db
from sqlalchemy import create_engine
import pandas as pd
import plotly.express as px
import datetime
import plotly.graph_objs as go
import warnings

DEFAULT_PLOTS = [
    "nena_px",
    "photons_mean",
    "frame_std",
    "n_locs",
    "locs_frame",
    "bg_mean",
    "drift_x",
    "drift_y",
]


@st.cache_data
def convert_df(df: pd.DataFrame):
    """
    Helper function to encode a dataframe as utf-8

    Args:
        df (pd.DataFrame): Dataframe to convert.
    """

    return df.to_csv().encode("utf-8")


def parse_input(input_: str):
    """
    Parses an input string and looks for & and |.
    Keywords with & will be added to an include list and keywords with | to an exclude list.

    Args:
        input_ (str): Input string
    """

    token = ("&", "|")

    mode = "add"

    last_idx = 0

    to_add = []
    to_exclude = []

    for idx, _ in enumerate(input_):

        if _ in token:
            if mode == "add":
                to_add.append(input_[last_idx:idx])
                last_idx = idx + 1
            else:
                to_exclude.append(input_[last_idx:idx])
                last_idx = idx + 1

            if _ == "&":
                mode = "add"
            elif _ == "|":
                mode = "exclude"

    # final cleanup
    if mode == "add":
        to_add.append(input_[last_idx:])
    else:
        to_exclude.append(input_[last_idx:])

    to_add = [_ for _ in to_add if len(_) > 0]

    return to_add, to_exclude


def filter_db(df_: pd.DataFrame):
    """
    Utility function to create a user interface to filter a dataframe.

    Args:
        df_ (pd.DataFrame): Input dataframe.
    """

    df = df_.copy()
    st.write("## Filter")
    c1, c2, c3 = st.columns((1, 1, 2))

    min_date = c1.date_input(
        "Minimum acquisition date",
        df["file_created"].min() - datetime.timedelta(days=1),
    )
    min_date = datetime.datetime.combine(min_date, datetime.datetime.min.time())

    max_date = c2.date_input(
        "Maximum acquisition date",
        df["file_created"].max() + datetime.timedelta(days=1),
    )
    max_date = datetime.datetime.combine(max_date, datetime.datetime.min.time())

    input = c3.text_input("Filter for tag in filename (use & and | to add and exclude)")

    input = input.replace(" ", "")

    if input != "":
        to_add, to_exclude = parse_input(input)
        if (len(to_add) > 0) & (len(to_exclude) > 0):
            st.text(
                f"Filtering for filenames containing {to_add} but not {to_exclude}."
            )

        elif (len(to_add)) > 0:
            st.text(f"Filtering for filenames containing {to_add}.")

        elif (len(to_exclude)) > 0:
            st.text(f"Filtering for filenames not containing {to_exclude}.")

        df = filter_by_tags(df, to_add, to_exclude)

    df = df[(df["file_created"] >= min_date) & (df["file_created"] <= max_date)]

    st.text(f"From {len(df_):,} Database entries, {len(df):,} are remaining.")

    return df


def filter_by_tags(df: pd.DataFrame, to_add: list, to_exclude: list):
    """
    Filters the entries of a dataframe according to an to_add and to_exclude list.
    Rows will be removed according to their filename.

    Args:
        df (pd.DataFrame): Dataframe
        to_add (list): keywords that should be in the filename.
        to_exclude (list): keywords that should not be in the filename.

    Returns:
        _type_: _description_
    """

    add = df["filename"].apply(lambda x: True if any(i in x for i in to_add) else False)

    if len(to_exclude) > 0:
        exclude = df["filename"].apply(
            lambda x: False if any(i in x for i in to_exclude) else True
        )
    else:
        exclude = [True for _ in range(len(df))]

    exclude = df["filename"].apply(
        lambda x: False if any(i in x for i in to_exclude) else True
    )

    return df[add & exclude]


def check_group(filename: str, groups: tuple):
    """Check if a filename belongs to a group

    Args:
        filename (str): filename
        groups (tuple): tuple with groups as strings
    """
    found = "None"
    for g in groups:
        if g in filename:
            found = g

    return found


def history():
    """
    Streamlit page to show the file history.
    """
    st.write("# History")

    df_ = fetch_db()

    if len(df_) > 0:
        options = df_.columns.tolist()
        options.remove("file_created")

        df = filter_db(df_)
        df = df.reset_index(drop=True)

        c1, c2 = st.columns(2)

        fields = c1.multiselect("Fields to plot", options, DEFAULT_PLOTS)
        groups = c2.text_input(
            "Enter tags to group (seperate by comma and no space)."
            "\nGroups will be color-coded in scatter and box plots."
        )
        groups_ = groups.split(",")

        df["group"] = df["filename"].apply(lambda x: check_group(x, groups_))

        df["file_created_date"] = df["file_created"].apply(lambda x: x.date())

        df = df.sort_values("file_created", ascending=False)

        c2.write(df["group"].value_counts())

        plotmode = st.selectbox("Plotmode", ["Table", "Scatter", "Box"])

        if plotmode == "Scatter":
            trendlines = st.checkbox("Show trendlines")
            if trendlines:
                trendline = "ols"
            else:
                trendline = None

        df["file_created_"] = df["file_created"].apply(lambda x: x.timestamp())
        df["file_created_date"] = df["file_created"].apply(lambda x: x.date())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with st.spinner("Creating plots.."):

                if plotmode == "Table":
                    table = df

                    if st.checkbox("Barchart in column"):
                        table_style = table.style.bar(color="lightgray")
                        st.write(
                            table_style.to_html(escape=False), unsafe_allow_html=True
                        )
                    else:
                        st.dataframe(table)

                    csv = convert_df(table)

                    st.download_button(
                        f"Click to download as (csv)",
                        csv,
                        "file.csv",
                        "text/csv",
                        key="download-csv",
                    )
                else:

                    for field in fields:
                        median_ = df[field].median()

                        if plotmode == "Scatter":
                            fig = px.scatter(
                                df,
                                x="file_created_",
                                y=field,
                                color="group",
                                hover_name="filename_hdf",
                                hover_data=["file_created"],
                                title=f"{field} - median {median_:.2f}",
                                trendline=trendline,
                                height=400,
                            )

                            fig.update_xaxes(
                                tickangle=45,
                                tickmode="array",
                                tickvals=df["file_created_"],
                                ticktext=df["file_created_date"],
                            )

                            st.plotly_chart(fig)
                        elif plotmode == "Box":
                            fig = px.box(
                                df,
                                x="file_created_date",
                                y=field,
                                color="group",
                                hover_name="filename_hdf",
                                hover_data=["file_created"],
                                title=f"{field} - median {median_:.2f}",
                                height=400,
                            )

                            st.plotly_chart(fig)

                        else:
                            pass
    else:
        st.warning("Database empty. Process files first.")
