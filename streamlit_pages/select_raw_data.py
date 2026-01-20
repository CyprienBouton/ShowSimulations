import streamlit as st
import os
import pickle
import MRzeroCore as mr0
import tempfile
from utils.RecoMRzero import RecoMRzero

def select_raw_data():
    st.title("Select Raw Data")
    
    if 'seq_file' in st.session_state:
        st.write(f"Current seq file: {st.session_state.seq_file}")
    uploaded_seq_file = st.file_uploader("Played sequence file .pkl or seq file", type=["pkl", "seq"])
    if uploaded_seq_file is not None:
        if uploaded_seq_file.name.endswith('.pkl'):
            # Save file to a temporary directory
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_seq_file.getbuffer())  # Save the contents of the uploaded file
                tmp_file_path = tmp_file.name  # Temporary file path
        
            with open(tmp_file_path, 'rb') as f:
                seq = pickle.load(f)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".seq") as tmp_file:
                tmp_file.write(uploaded_seq_file.getbuffer())  # Save file contents
                tmp_file_path = tmp_file.name  # Temporary file path
                seq = mr0.Sequence.import_file(tmp_file_path)
    
        reco = RecoMRzero(seq)
        st.session_state.df = reco.get_reco_dataframe()
        # Optionally, remove the temporary file if no longer needed
        os.remove(tmp_file_path)