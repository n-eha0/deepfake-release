import numpy as np
import plotly.graph_objects as go

def create_waveform_plot(y, sr=22050, max_points=20000):
    """
    Create waveform plot with proper downsampling and visibility boost.
    """
    if y is None or len(y) == 0:
        y = np.zeros(sr)

    # Amplify quiet signals for visibility
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))  # normalize
        y = y * 0.9  # keep within [-1, 1]

    # Downsample if too long
    if len(y) > max_points:
        step = int(len(y) / max_points)
        y = y[::step]
        time = np.arange(len(y)) * step / sr
    else:
        time = np.linspace(0, len(y) / sr, num=len(y))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=y,
        mode="lines",
        line=dict(color="royalblue", width=1)
    ))
    fig.update_layout(
        title="Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=300,
        margin={'t':30,'b':20,'l':40,'r':20},
        showlegend=False
    )
    return fig

def create_confidence_meter(confidence, threshold=50.0):
    """
    Fun & modern confidence gauge:
    - Neon colors (green → yellow → red).
    - Dark background.
    - Bold glowing style.
    """
    steps = [
        {'range': [0, threshold*0.5], 'color': "#00FF7F"},     # neon green
        {'range': [threshold*0.5, threshold], 'color': "#FFD700"},  # gold
        {'range': [threshold, 100], 'color': "#FF1493"}        # neon pink/red
    ]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Deepfake Confidence", 'font': {'size': 22, 'color': '#FFFFFF'}},
        number={'suffix': " %", 'font': {'size': 32, 'color': '#00E5FF'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#888"},
            'bar': {'color': "#1E90FF", 'thickness': 0.25},  # neon blue bar
            'bgcolor': "#111",
            'steps': steps,
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))

    fig.update_layout(
        height=400,
        margin={'t':50, 'b':20, 'l':20, 'r':20},
        paper_bgcolor="#000",   # dark theme
        font={'color': "#EEE", 'family': "Courier New"}
    )
    
    return fig
