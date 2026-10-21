import os
import sqlite3
from contextlib import closing

import pandas as pd
import streamlit as st
import time
from picasso.localize import db_filename

#: the columns of the tables of the database that hold dates and times
#: (SQLite stores them as text)
DATE_COLUMNS = {
    "files": ["file_created", "entry_created"],
    "watcher": ["started"],
}


def read_table(name: str) -> pd.DataFrame:
    """Read a table of the local database (see ``db_filename``).

    Parameters
    ----------
    name : str
        ``"files"`` or ``"watcher"``.

    Returns
    -------
    df : pd.DataFrame
        The table, with its dates and times as datetimes; empty if it
        does not exist yet.
    """
    with closing(sqlite3.connect(db_filename())) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        if exists is None:
            return pd.DataFrame()
        return pd.read_sql_query(
            f'SELECT * FROM "{name}"',
            connection,
            # with or without fractions of a second
            parse_dates={
                column: {"format": "ISO8601"}
                for column in DATE_COLUMNS.get(name, [])
            },
        )


def write_table(df: pd.DataFrame, name: str, if_exists: str) -> None:
    """Write ``df`` to a table of the local database (see
    ``db_filename``).

    Parameters
    ----------
    df : pd.DataFrame
        The rows.
    name : str
        The table.
    if_exists : {"append", "replace"}
        Add the rows to the table or replace it.
    """
    with closing(sqlite3.connect(db_filename())) as connection, connection:
        df.to_sql(name, con=connection, if_exists=if_exists, index=False)


def fetch_db():
    """Load the local database and return the localized files.

    Returns
    -------
    df : pd.DataFrame
        The ``files`` table, sorted by acquisition date; empty if the table
        does not exist yet.
    """
    df = read_table("files")
    if len(df):
        df = df.sort_values("file_created")
    return df


def get_file_family(path: str) -> tuple[list[str], str]:
    """Find the localization files that belong to a movie.

    Picasso names the files it derives from a movie after the movie
    without its extension, followed by ``_`` (e.g., ``movie_locs.hdf5``
    and ``movie_locs_undrift.hdf5`` for ``movie.raw``, or
    ``Pos0.ome_locs.hdf5`` for ``Pos0.ome.tif``). Files of other movies
    whose names merely start the same (``movie10_locs.hdf5``) are not
    part of the family.

    Parameters
    ----------
    path : str
        Path to the movie (or to a localization file).

    Returns
    -------
    files : list of str
        Names of the ``.hdf5`` files of the family, sorted.
    folder : str
        The folder they are in.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist.
    """
    folder = os.path.dirname(path)
    base = os.path.splitext(os.path.basename(path))[0]
    files = sorted(
        name
        for name in os.listdir(folder or ".")
        if name.endswith(".hdf5")
        and (name == base + ".hdf5" or name.startswith(base + "_"))
    )
    return files, folder


def fetch_watcher():
    """Load the local database and return the running watchers.

    Returns
    -------
    df : pd.DataFrame
        The ``watcher`` table; empty if it does not exist yet.
    """
    return read_table("watcher")


def refresh(to_wait: int):
    """Wait for a given amount of time and then stop streamlit.

    Parameters
    ----------
    to_wait : int
        Seconds to count down before rerunning the page.
    """
    ref = st.empty()
    for i in range(to_wait):
        ref.write(f"Refreshing in {to_wait-i} s")
        time.sleep(1)
    st.stop()
