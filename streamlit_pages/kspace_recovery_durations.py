import streamlit as st
import io
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def plot_fig(df, marker_size, is3D, show_flags, cmin, cmax):
    # Set colorbar scale
    if cmin is None:
        cmin = df.RD.min()
    if cmax is None:
        cmax = df.RD.max()

    y = df.Par
    ylabel = 'Partition' if is3D else 'Slice'
    
    # Prepare hover data
    base_customdata = np.vstack([[ylabel] * len(df), df.RD]).T
    customdata = base_customdata if not show_flags \
        else np.hstack([base_customdata, df[['Flags']].values])
    hovertemplate = (
        f'Line: %{{x}}<br>{ylabel}: %{{y}}<br>recovery duration: %{{customdata[1]:.2f}} s' +
        ('<br>Flags: %{customdata[2]}' if show_flags else '') +
        '<extra></extra>'
    )


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.Lin,
        y=y,
        mode='markers',
        marker=dict(size=marker_size, color=df.RD, colorscale='jet',
                    colorbar=dict(title='recovery duration (s)'),
                    cmin=cmin, cmax=cmax),
        customdata=customdata,
        hovertemplate=hovertemplate,
        showlegend=False,
    ))
    fig.update_layout(
        xaxis=dict(title="Line"),
        yaxis=dict(title=ylabel),
        height=800, width=800,
        template='simple_white'
    )
    return fig

def kspace_recovery_durations():
    st.header("K-Space Recovery Durations Map")
    if 'df' not in st.session_state:
        st.error("❗ Please upload a raw data file  first.")
        return
    
    df = st.session_state.df

    is3D = True

    marker_size = st.sidebar.slider("Marker Size", 2, 10, 6)
    show_flags = st.sidebar.checkbox("Show Flags", value=False)

    scale_colorbar = st.sidebar.checkbox("Scale colorbar", value=True)

    if scale_colorbar:
        cmin = st.sidebar.slider("Colorbar Min (s)", 0.0, 5.0, 0.4, step=0.1)
        cmax = st.sidebar.slider("Colorbar Max (s)", 0.5, 10.0, 2.0, step=0.1)
    else:
        cmin, cmax = None, None


    fig = plot_fig(df, marker_size, is3D, show_flags, cmin, cmax)
    st.plotly_chart(fig, use_container_width=True)
