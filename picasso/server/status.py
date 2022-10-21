import streamlit as st
from helper import fetch_db
from picasso import localize
import pandas as pd
from sqlalchemy import create_engine
import os


def check_file(file):
    base, ext = os.path.splitext(file)
    file_hdf = base + "_locs.hdf5"

    return os.path.isfile(file_hdf)


def escape_markdown(text: str) -> str:
    """Helper function to escape markdown in text.
    Args:
        text (str): Input text.
    Returns:
        str: Converted text to be used in markdown.
    """
    MD_SPECIAL_CHARS = "\`*_{}[]()#+-.!"
    for char in MD_SPECIAL_CHARS:
        text = text.replace(char, "\\" + char)
    return text


def status():
    """
    Streamlit page to show the status page.
    """
    st.write("# Status")

    with st.expander("Getting started"):
        st.write(
            f"Picasso server allows to monitor perfomance of your super resolution runs. By selecting `Estimate and add to database` in localize, summary statistics of a run will be stored in a local database in the picasso user folder ({escape_markdown(localize._db_filename())})."
        )
        st.write(
            "- Status: Displays the current database status and documentation."
            " \n- History: Explore summary statistics of processed files."
            " \n- Compare: Compare two files against each other."
            " \n- Watcher: Set up a file watcher to process files automatically."
            " \n- Preview: Preview will render the super-resolution data in the browser."
        )

    with st.expander("Database overview"):
        st.write(
            "If you want to read and modify the database directly use tools like [DB Browser](https://sqlitebrowser.org/)."
        )
        df = fetch_db()
        if len(df) > 0:
            df = df.sort_values("entry_created")
            st.write(f"The database currently contains {len(df):,} entries.")
            st.write("Preview of the last 10 entries:")
            st.write(
                df.iloc[-10:][["entry_created", "filename", "nena_px", "file_created"]]
            )
        else:
            df = pd.DataFrame(
                columns=["entry_created", "filename", "nena_px", "file_created"]
            )
            st.write("Database is empty.")

    with st.expander("Manually add file to database."):
        st.write(
            "Here, you can manually add files to the database."
            " \n- Enter the path of a image stack (`.raw`, `.ome.tif`) or a folder with multiple image stacks."
            " \n- All files that were reconstructed (i.e. have a `_locs.hdf5`-file) will be considered ."
            " \n- Drift will only be added if a undrifted file `_undrift.hdf5` is present."
            " \n- Files that are already in the database will be ignored."
            " \n- Consectuive files (`Pos0.ome.tif`, `Pos0_1.ome.tif`, `Pos0_2.ome.tif`) will be treated as one."
        )
        path = st.text_input("Enter file path or folder:")

        if check_file(path) & path.endswith((".raw", ".ome.tif", ".ims")):

            if path not in df["filename"].tolist():
                base, ext = os.path.splitext(path)
                target = base + '_locs.hdf5'

                if not os.path.isfile(target):
                    st.error(f"File {target} does not exist.")
                else:
                    file_hdf = target

                with st.spinner(f"Fetching summary from {file_hdf}."):
                    summary = localize.get_file_summary(path, file_hdf=file_hdf)
                    st.write(summary)
                    if st.button("Add to database"):
                        engine = create_engine(
                            "sqlite:///" + localize._db_filename(), echo=False
                        )
                        pd.DataFrame(summary.values(), summary.keys()).T.to_sql(
                            "files", con=engine, if_exists="append", index=False
                        )
                        st.success("Submitted to DB. Please refresh page.")
            else:
                st.error("File already in database.")

        elif os.path.isdir(path):
            files = [
                _ for _ in os.listdir(path) if _.endswith((".raw", ".ome.tif", ".ims"))
            ]  # Children files are in there

            n_files = len(files)
            st.text(f"A total of {n_files} files in folder.")

            pbar = st.progress(0)

            if st.button("Add files"):
                current_file = st.empty()
                all_df = []
                for idx, file in enumerate(files):
                    current_file.text(f"Current file {file}.")
                    path_ = os.path.join(path, file)
                    if path_ not in df["filename"].tolist():
                        base, ext = os.path.splitext(path_)
                        file_hdf = base + "_locs.hdf5"
                        if os.path.isfile(file_hdf):
                            summary = localize.get_file_summary(path_, file_hdf=file_hdf)
                            df_ = pd.DataFrame(summary.values(), summary.keys()).T
                            all_df.append(df_)
                        else:
                            st.error(f"File {target} does not exist.")
                    pbar.progress(int((idx + 1) / (n_files) * 100))

                if len(all_df) > 0:
                    stack = pd.concat(all_df)

                    st.write(stack)


                    engine = create_engine(
                        "sqlite:///" + localize._db_filename(), echo=False
                    )
                    stack.to_sql("files", con=engine, if_exists="append", index=False)

                    st.success(f"Submitted {len(stack)} entries to the DB.")
                    st.success("Submitted to DB. Please refresh page.")
                else:
                    st.warning('No files found in folder.')
        else:
            st.warning(f"Path is not valid or no locs found.")
