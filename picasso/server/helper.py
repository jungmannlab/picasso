import picasso.io
import picasso.postprocess
import os
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd
import streamlit as st
import time
import subprocess
import picasso.localize
from picasso.localize import _db_filename


def fetch_db():
    """
    Helper function to load the local database and return the files.
    """
    try:
        DB_PATH = "sqlite:///" + _db_filename()
        engine = create_engine(DB_PATH, echo=False)
        df = pd.read_sql_table("files", con=engine)

        df = df.sort_values("file_created")
    except ValueError:
        df = pd.DataFrame()

    return df


def fetch_watcher():
    """
    Helper function to load the local database and return running watchers.
    """
    try:
        engine = create_engine("sqlite:///" + _db_filename(), echo=False)
        df = pd.read_sql_table("watcher", con=engine)
    except ValueError as e:
        print(e)
        df = pd.DataFrame()

    return df


def refresh(to_wait: int):
    """
    Utility function that waits for a given amount and then stops streamlit.
    """
    ref = st.empty()
    for i in range(to_wait):
        ref.write(f"Refreshing in {to_wait-i} s")
        time.sleep(1)
    st.stop()
