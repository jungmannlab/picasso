"""
Streamlit application to interface with the database
"""

import os
import socket
from PIL import Image
import streamlit as st
from sqlalchemy import create_engine

from picasso import localize
from status import status
from preview import preview
from history import history
from watcher import watcher
from compare import compare

from picasso import __version__ as VERSION_NO

_this_file = os.path.abspath(__file__)
_this_directory = os.path.dirname(_this_file)

LOGO_PATH = os.path.abspath(
    os.path.join(_this_directory, os.pardir, "gui/icons/picasso_server.png")
)
logo = Image.open(LOGO_PATH)

st.set_page_config(page_title="Picasso Server", page_icon=logo, layout="wide")

c1, c2, c3, c4 = st.sidebar.columns((1, 1, 1, 1))
c1.image(logo)
c2.write("# Picasso Server")

engine = create_engine("sqlite:///" + localize.db_filename(), echo=False)

st.sidebar.code(f"{socket.gethostname()}\nVersion {VERSION_NO}")

sidebar = {
    "Status": status,
    "History": history,
    "Compare": compare,
    "Watcher": watcher,
    "Preview": preview,
}

# Material Symbols, bundled with Streamlit
icons = {
    "Status": ":material/monitor_heart:",
    "History": ":material/history:",
    "Compare": ":material/compare_arrows:",
    "Watcher": ":material/visibility:",
    "Preview": ":material/image:",
}

menu = st.sidebar.radio(
    "Page",
    list(sidebar.keys()),
    format_func=lambda page: f"{icons[page]} {page}",
    label_visibility="collapsed",
)

if menu:
    sidebar[menu]()
