"""
Streamlit application to interface with the database
"""

import os
import socket
from PIL import Image
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

from picasso import localize
from status import status
from preview import preview
from history import history
from watcher import watcher
from compare import compare

from picasso.__version__ import VERSION_NO

st.set_page_config(layout="wide")


_this_file = os.path.abspath(__file__)
_this_directory = os.path.dirname(_this_file)

LOGO_PATH = os.path.abspath(
    os.path.join(_this_directory, os.pardir, "gui/icons/picasso_server.png")
)
logo = Image.open(LOGO_PATH)

c1, c2, c3, c4 = st.sidebar.columns((1, 1, 1, 1))
c1.image(logo)
c2.write("# Picasso Server")

engine = create_engine("sqlite:///" + localize._db_filename(), echo=False)

st.sidebar.code(f"{socket.gethostname()}\nVersion {VERSION_NO}")

sidebar = {
    "Status": status,
    "History": history,
    "Compare": compare,
    "Watcher": watcher,
    "Preview": preview,
}

menu = st.sidebar.radio("", list(sidebar.keys()))

if menu:
    sidebar[menu]()
