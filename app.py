import streamlit as st
import pandas as pd 
try:
    import ydata_profiling as pdp
except ImportError:
    import pandas_profiling as pdp
from streamlit_pandas_profiling import st_profile_report
import os
from pycaret.classification import setup, compare_models, pull, save_model


if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'refreshed' not in st.session_state:
    st.session_state.refreshed = False
if 'profile_report' not in st.session_state:
    st.session_state.profile_report = None

def clear_data_on_refresh():
    if st.session_state.data_loaded and st.session_state.refreshed:
        # Clear session state data
        st.session_state.data_loaded = False
        st.session_state.refreshed = False
        st.session_state.profile_report = None

        # Delete CSV file
        if os.path.exists("sourcedata.csv"):
            os.remove("sourcedata.csv")

# Function to display a warning if no data is loaded
def display_data_warning():
    if not st.session_state.data_loaded:
        st.warning("Please upload data first.")

# Perform data clearing on page refresh
clear_data_on_refresh()

with st.sidebar:
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)
else:
    df = pd.DataFrame()  

if choice == "Upload":
    st.title("Upload Data for Modelling")
    file = st.file_uploader("Upload your dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
        st.session_state.data_loaded = True

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    
    if st.session_state.data_loaded and st.session_state.profile_report is None:
        st.session_state.profile_report = pdp.ProfileReport(df)
    
    if st.session_state.data_loaded and st.session_state.profile_report is not None:
        st_profile_report(st.session_state.profile_report)
            
    elif not st.session_state.data_loaded:
        display_data_warning()

if choice == "ML":
    st.title("ML Operations")
    target = st.selectbox("Select Target Variable", df.columns)
    problem_type = st.radio("Select Problem Type", ["classification", "regression"])

    if st.session_state.data_loaded and st.button("Setup Experiment"):
        try:
            setup_df = setup(df, target=target, session_id=123, categorical_features=[], numeric_features=[], ignore_features=[])

            if problem_type == "classification":
                best_model = compare_models()
            elif problem_type == "regression":
                best_model = compare_models(fold=5, round=2, sort='R2')

            compare_df = pull()
            st.info("Best Models: ")
            st.dataframe(compare_df)
            st.write("Best Model:")
            st.write(best_model)
            save_model(best_model, "best_model")  
        except ValueError as e:
            st.error(str("Invalid Selction: Please select either problem"))
    elif not st.session_state.data_loaded:
        display_data_warning()

def handle_page_refresh():
    if st.button("Refresh Page"):
        st.session_state.refreshed = True
        st.experimental_rerun()
handle_page_refresh()

if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download Model", f, "trained_model.pkl")
