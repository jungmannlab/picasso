import picasso.io
import picasso.postprocess
import os
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
import streamlit as st
import time


def _db_filename():
    home = os.path.expanduser("~")
    return os.path.abspath(os.path.join(home, ".picasso", "app.db"))


def fetch_db():
    try:
        engine = create_engine("sqlite:///" + _db_filename(), echo=False)
        df = pd.read_sql_table("files", con=engine)

        df = df.sort_values("file_created")
    except ValueError:
        df = pd.DataFrame()

    return df


def fetch_watcher():
    try:
        engine = create_engine("sqlite:///" + _db_filename(), echo=False)
        df = pd.read_sql_table("watcher", con=engine)
    except ValueError:
        df = pd.DataFrame()

    return df


def refresh(to_wait: int):
    """
    Utility function that waits for a given amount and then restarts streamlit.
    """
    ref = st.empty()
    for i in range(to_wait):
        ref.write(f"Refreshing in {to_wait-i} s")
        time.sleep(1)
    raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))
