import streamlit as st
from helper import _db_filename
from sqlalchemy import create_engine
import pandas as pd
import os
import numpy as np
from picasso import io
from picasso import render
import matplotlib.pyplot as plt


@st.cache_data
def load_file(path: str):
    """Loads localization from files. Cached version.

    Args:
        path (str): Path to file.
    """
    locs, info = io.load_locs(path)
    return locs, info


@st.cache_data
def picasso_render(locs: np.ndarray, viewport: tuple, oversampling: float):
    """Helper function to render a viewport. Cached.

    Args:
        locs (np.ndarray): Record array with localization data.
        viewport (tuple): Viewport as tuple.
        oversampling (int): Oversampling.
    """
    len_x, image = render.render(
        locs,
        viewport=viewport,
        oversampling=oversampling,
        blur_method="smooth",
    )

    return image


def preview():
    """
    Streamlit page to preview a file.
    """
    st.write("# Preview")

    st.write(
        "Select a movie from the database. All hdf files with the same base path as the movie will be selectable."
    )
    engine = create_engine("sqlite:///" + _db_filename(), echo=False)

    try:
        df = pd.read_sql_table("files", con=engine)

        file = st.selectbox("Select file", df["filename"].unique())
        # Find files in familiy
        base = os.path.split(file)[1].split(".")[0]
        folder = os.path.dirname(file)

        if os.path.isdir(folder):

            files = os.listdir(folder)
            files = [f for f in files if f.startswith(base) and f.endswith(".hdf5")]

            hdf_file = st.selectbox("Select hdf file", [None] + files)

            if hdf_file is not None:
                st.write("## File preview")

                st.info(
                    "Performance Warning: This preview will render the full image, so it might be slow for large oversmapling."
                )

                with st.spinner("Loading file"):
                    hdf_file_ = os.path.join(folder, hdf_file)
                    if os.path.isfile(hdf_file_):
                        locs, info = load_file(hdf_file_)

                        x_min = np.min(locs.x)
                        x_max = np.max(locs.x)
                        y_min = np.min(locs.y)
                        y_max = np.max(locs.y)

                        viewport = (y_min, x_min), (y_max, x_max)

                        c1, c2, c3 = st.columns(3)

                        oversampling = c1.number_input(
                            "Oversampling", value=5.0, min_value=1., max_value=40.
                        )

                        image = picasso_render(locs, viewport, oversampling)

                        vmin = c2.number_input(
                            "Min density", value=np.min(image.flatten())
                        )
                        vmax = c3.number_input(
                            "Max density", value=np.max(image.flatten())
                        )

                        # plt.imshow(image, cmap='hot', vmax=10)
                        fig, ax = plt.subplots()
                        st.write(f"Image with dimensions {image.shape}")
                        im = ax.imshow(image, cmap="hot", vmin=vmin, vmax=vmax)
                        # Hide grid lines
                        ax.grid(False)

                        # Hide axes ticks
                        ax.set_xticks([])
                        ax.set_yticks([])
                        st.pyplot(fig)

                        st.write(df[df["filename"] == file].iloc[0].to_dict())

                    else:
                        st.warning(f"File {hdf_file} not found.")
        else:
            st.error(
                f"Couldn't find folder {folder}. Please check if folder was deleted or moved."
            )

    except ValueError:
        st.write("Database empty. Process files first.")
