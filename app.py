"""ShowSimulations: Streamlit app to inspect the timing of MRzero (simulated) sequences.

Run with ``streamlit run app.py``. Start on the "Choose raw data" page to
upload a played sequence; the other pages read ``st.session_state.df`` /
``st.session_state.reco``.
"""
import streamlit as st

from streamlit_pages import (
    kspace_recovery_durations,
    kspace_timing_map,
    longitudinal_magnetizations,
    recovery_durations_stats,
    select_raw_data,
)

if __name__ == "__main__":
    page_names_to_funcs = {
        "Choose raw data": select_raw_data.select_raw_data,
        "Acquisition Timeline": kspace_timing_map.kspace_timing_map,
        "Recovery Durations": kspace_recovery_durations.kspace_recovery_durations,
        "Recovery Durations Stats": recovery_durations_stats.recovery_durations_stats,
        "Longitudinal Magnetizations": longitudinal_magnetizations.longitudinal_magnetizations,
    }
    selected_page = st.sidebar.selectbox("Go to page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()
