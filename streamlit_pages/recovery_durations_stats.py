import streamlit as st
import io
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def plot_hist(df, rd_min=None, rd_max=None):
    fig = go.Figure()
    if rd_min is not None and rd_max is not None:
        x_range = rd_max - rd_min
    else:
        x_range = df.RD.max() - df.RD.min()

    nbinsx = int(50 * (df.RD.max() - df.RD.min())/ x_range)
    
    # Adjusting the number of bins to control bar width
    fig.add_trace(go.Histogram(
        x=df.RD, 
        nbinsx=nbinsx  # Fewer bins = wider bars
    ))

    fig.update_layout(
        xaxis=dict(
            title="Recovery duration (seconds)",
            title_font=dict(size=20),  # Title font size
            tickfont=dict(size=20),    # Tick label font size
            range=[rd_min, rd_max] if rd_min is not None and rd_max is not None else None
        ),
        yaxis=dict(
            title="Number of occurrences",
            title_font=dict(size=20),  # Title font size
            tickfont=dict(size=20)     # Tick label font size
        ),
        height=800, width=800,
        template='simple_white',
        bargap=0.1,  # Slight gap between bars
    )
    
    return fig

def recovery_durations_stats():
    st.header("K-Space Recovery Durations Statistics")
    if 'df' not in st.session_state:
        st.error("❗ Please upload a raw data file  first.")
        return
    
    df = st.session_state.df
    
    # Sidebar controls for scaling
    scale_hist = st.sidebar.checkbox("Scale x-axis (RD)", value=True)

    if scale_hist or np.ptp(df.RD)==0: # if all RD values are the same, allow scaling to visualize the histogram
        rd_min = st.sidebar.slider("RD Min (s)", 0.0, 5.0, 0.4, step=0.1)
        rd_max = st.sidebar.slider("RD Max (s)", 0.5, 10.0, 2.0, step=0.1)
    else:
        rd_min, rd_max = None, None

    fig = plot_hist(df, rd_min, rd_max)
    st.plotly_chart(fig, use_container_width=True)
