"""Page "Choose raw data": upload the played MRzero sequence (.pkl or .seq).

Stores in ``st.session_state``: ``reco`` (:class:`utils.RecoMRzero.RecoMRzero`),
``df`` (one row per k-space line: Lin, Par, Time, RD) and ``seq_file``.
"""
import os
import pickle
import tempfile

import MRzeroCore as mr0
import streamlit as st

from utils.RecoMRzero import RecoMRzero


def load_sequence(uploaded_seq_file):
    """Return the MRzero sequence of an uploaded pickled sequence (.pkl) or Pulseq file (.seq)."""
    if uploaded_seq_file.name.endswith('.pkl'):
        return pickle.loads(uploaded_seq_file.getbuffer())
    # Pulseq import needs a file on disk; close (flush) it before reading it back.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".seq") as tmp_file:
        tmp_file.write(uploaded_seq_file.getbuffer())
        tmp_file_path = tmp_file.name
    try:
        return mr0.Sequence.import_file(tmp_file_path)
    finally:
        os.remove(tmp_file_path)


def select_raw_data():
    st.title("Select Raw Data")

    if 'seq_file' in st.session_state:
        st.write(f"Current seq file: {st.session_state.seq_file}")
    uploaded_seq_file = st.file_uploader("Played sequence file .pkl or seq file", type=["pkl", "seq"])
    if uploaded_seq_file is not None:
        seq = load_sequence(uploaded_seq_file)
        reco = RecoMRzero(seq)
        st.session_state.reco = reco
        st.session_state.df = reco.get_reco_dataframe()
        st.session_state.seq_file = uploaded_seq_file.name
        st.success("File loaded successfully!")
