import streamlit as st
from helper import get_file_summary
from datetime import datetime
import os
import pandas as pd


def inspect():

    st.write("# Inspect ")

    # Select file from database

    file = st.text_input("Enter file path:")

    if file:
        sum_ = get_file_summary(file)

        sum_["filename"] = file
        sum_["file_created"] = datetime.fromtimestamp(os.path.getmtime(file))

        # File processed
        s = pd.Series(sum_, index=sum_.keys()).to_frame().T
        st.table(pd.DataFrame(s))

        if st.button("Submit"):
            s.to_sql("files", con=engine, if_exists="append", index=False)
