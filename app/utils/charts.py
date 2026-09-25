"""
Plotly Chart Builders for Financial Market Intelligence Engine
Configured with dark financial theme tokens.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from app.utils.theme import (
    COLOR_BG, COLOR_SURFACE, COLOR_BORDER, COLOR_BORDER_LIGHT,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED,
    COLOR_ACCENT_BLUE, COLOR_ACCENT_GREEN, COLOR_ACCENT_RED, COLOR_ACCENT_AMBER, COLOR_ACCENT_PURPLE,
    get_plotly_dark_layout
)

STATE_COLORS = {
    0: COLOR_ACCENT_GREEN,   # State 0: Low-Vol Bull
    1: COLOR_ACCENT_RED,     # State 1: High-Vol Bear
    2: COLOR_ACCENT_AMBER,   # State 2: Volatile Recovery
    3: COLOR_ACCENT_BLUE     # State 3: Consolidation
}

STATE_LABELS = {
    0: "State 0 (Low-Vol Bull)",
    1: "State 1 (High-Vol Bear)",
    2: "State 2 (Volatile Recovery)",
    3: "State 3 (Consolidation)"
}


def create_sp500_price_chart(df: pd.DataFrame) -> go.Figure:
    """
    Creates a Plotly line chart of S&P 500 price history.
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="S&P 500 (^GSPC)",
            line=dict(color=COLOR_ACCENT_BLUE, width=2),
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Close</b>: $%{y:,.2f}<extra></extra>"
        )
    )
    
    layout = get_plotly_dark_layout()
    layout.update({
        "title": {"text": "S&P 500 Index Price History (^GSPC)", "x": 0.0},
        "xaxis": {**layout["xaxis"], "title": "Date"},
        "yaxis": {**layout["yaxis"], "title": "Index Level ($)", "tickprefix": "$", "tickformat": ",.0f"},
        "height": 380,
        "showlegend": False
    })
    
    fig.update_layout(layout)
    return fig


def create_daily_returns_chart(df: pd.DataFrame) -> go.Figure:
    """
    Creates a Plotly bar chart of daily percentage returns with positive/negative color coding.
    """
    returns_pct = df["Daily_Return"] * 100
    colors = [COLOR_ACCENT_GREEN if val >= 0 else COLOR_ACCENT_RED for val in returns_pct]
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=returns_pct,
            marker_color=colors,
            name="Daily Return (%)",
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Return</b>: %{y:+.2f}%<extra></extra>"
        )
    )
    
    layout = get_plotly_dark_layout()
    layout.update({
        "title": {"text": "S&P 500 Daily Returns (%)", "x": 0.0},
        "xaxis": {**layout["xaxis"], "title": "Date"},
        "yaxis": {**layout["yaxis"], "title": "Daily Return (%)", "ticksuffix": "%"},
        "height": 260,
        "showlegend": False
    })
    
    fig.update_layout(layout)
    return fig


def create_rolling_volatility_chart(df: pd.DataFrame) -> go.Figure:
    """
    Creates a Plotly line chart of 20-day rolling volatility.
    """
    vol_pct = df["Rolling_Volatility_20"] * 100
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=vol_pct,
            mode="lines",
            name="20D Volatility (%)",
            line=dict(color=COLOR_ACCENT_AMBER, width=2),
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>20D Volatility</b>: %{y:.2f}%<extra></extra>"
        )
    )
    
    layout = get_plotly_dark_layout()
    layout.update({
        "title": {"text": "20-Day Rolling Daily Volatility (%)", "x": 0.0},
        "xaxis": {**layout["xaxis"], "title": "Date"},
        "yaxis": {**layout["yaxis"], "title": "Volatility (%)", "ticksuffix": "%"},
        "height": 260,
        "showlegend": False
    })
    
    fig.update_layout(layout)
    return fig


def create_drawdown_chart(df: pd.DataFrame) -> go.Figure:
    """
    Creates a Plotly area chart of peak-to-trough drawdown depth.
    """
    dd_pct = df["Drawdown"] * 100
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=dd_pct,
            mode="lines",
            name="Drawdown (%)",
            fill="tozeroy",
            line=dict(color=COLOR_ACCENT_RED, width=1.5),
            fillcolor="rgba(248, 81, 73, 0.2)",
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Drawdown</b>: %{y:.2f}%<extra></extra>"
        )
    )
    
    layout = get_plotly_dark_layout()
    layout.update({
        "title": {"text": "S&P 500 Peak-to-Trough Drawdown Depth (%)", "x": 0.0},
        "xaxis": {**layout["xaxis"], "title": "Date"},
        "yaxis": {**layout["yaxis"], "title": "Drawdown (%)", "ticksuffix": "%"},
        "height": 260,
        "showlegend": False
    })
    
    fig.update_layout(layout)
    return fig


def create_cross_asset_correlation_chart(df: pd.DataFrame) -> go.Figure:
    """
    Creates a Plotly multi-line chart for 20-day S&P 500 correlations with TLT and GLD.
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SP500_TLT_Corr_20"],
            mode="lines",
            name="S&P 500 ↔ TLT (Treasuries)",
            line=dict(color=COLOR_ACCENT_BLUE, width=1.8),
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>S&P 500 ↔ TLT</b>: %{y:+.3f}<extra></extra>"
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SP500_GLD_Corr_20"],
            mode="lines",
            name="S&P 500 ↔ GLD (Gold)",
            line=dict(color=COLOR_ACCENT_AMBER, width=1.8),
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>S&P 500 ↔ GLD</b>: %{y:+.3f}<extra></extra>"
        )
    )
    
    fig.add_shape(
        type="line",
        x0=df.index.min(),
        x1=df.index.max(),
        y0=0,
        y1=0,
        line=dict(color=COLOR_BORDER_LIGHT, width=1, dash="dash")
    )
    
    layout = get_plotly_dark_layout()
    layout.update({
        "title": {"text": "20-Day Rolling Cross-Asset Correlations", "x": 0.0},
        "xaxis": {**layout["xaxis"], "title": "Date"},
        "yaxis": {**layout["yaxis"], "title": "Correlation (-1 to +1)", "range": [-1.05, 1.05]},
        "height": 320,
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"color": COLOR_TEXT_SECONDARY, "size": 11}
        }
    })
    
    fig.update_layout(layout)
    return fig


def create_hmm_regime_timeline_chart(df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive Plotly visualization showing S&P 500 price history
    color-coded by decoded 4-State Gaussian HMM market regime states.
    """
    fig = go.Figure()
    
    for state in range(4):
        state_mask = df["HMM_State"] == state
        state_df = df[state_mask]
        
        if state_df.empty:
            continue
            
        fig.add_trace(
            go.Scatter(
                x=state_df.index,
                y=state_df["Close"],
                mode="markers",
                name=STATE_LABELS[state],
                marker=dict(
                    color=STATE_COLORS[state],
                    size=4,
                    opacity=0.85
                ),
                customdata=np.stack((
                    state_df["HMM_State"],
                    state_df["Daily_Return"] * 100,
                    state_df["Rolling_Volatility_20"] * 100,
                    state_df["Momentum_20"] * 100,
                    state_df["Drawdown"] * 100
                ), axis=-1),
                hovertemplate=(
                    "<b>Date</b>: %{x|%Y-%m-%d}<br>"
                    "<b>HMM State</b>: State %{customdata[0]:.0f}<br>"
                    "<b>S&P 500 Close</b>: $%{y:,.2f}<br>"
                    "<b>Daily Return</b>: %{customdata[1]:+.2f}%<br>"
                    "<b>20D Volatility</b>: %{customdata[2]:.2f}%<br>"
                    "<b>20D Momentum</b>: %{customdata[3]:+.2f}%<br>"
                    "<b>Drawdown</b>: %{customdata[4]:.2f}%<extra></extra>"
                )
            )
        )
        
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="S&P 500 Trajectory",
            line=dict(color=COLOR_BORDER, width=1),
            showlegend=False,
            hoverinfo="skip"
        )
    )
    
    layout = get_plotly_dark_layout()
    layout.update({
        "title": {"text": "Decoded HMM Market Regime Timeline (S&P 500)", "x": 0.0},
        "xaxis": {**layout["xaxis"], "title": "Date"},
        "yaxis": {**layout["yaxis"], "title": "Index Level ($)", "tickprefix": "$", "tickformat": ",.0f"},
        "height": 420,
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"color": COLOR_TEXT_SECONDARY, "size": 11}
        }
    })
    
    fig.update_layout(layout)
    return fig


def create_transition_matrix_heatmap(transmat: np.ndarray) -> go.Figure:
    """
    Creates a Plotly Heatmap of the fitted 4x4 HMM state transition probability matrix.
    """
    labels = ["State 0", "State 1", "State 2", "State 3"]
    text_matrix = [[f"{val:.4f}" for val in row] for row in transmat]
    
    fig = go.Figure(
        data=go.Heatmap(
            z=transmat,
            x=[f"To {l}" for l in labels],
            y=[f"From {l}" for l in labels],
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 12, "color": COLOR_TEXT_PRIMARY},
            colorscale=[
                [0.0, "#0D1117"],
                [0.1, "#161B22"],
                [0.5, "#1F6FEB"],
                [1.0, "#58A6FF"]
            ],
            showscale=False,
            hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>Probability: %{z:.4f}<extra></extra>"
        )
    )
    
    layout = get_plotly_dark_layout()
    layout.update({
        "title": {"text": "HMM Transition Probability Matrix (A = [a_ij])", "x": 0.0},
        "xaxis": {**layout["xaxis"], "title": "Next State (t+1)"},
        "yaxis": {**layout["yaxis"], "title": "Current State (t)", "autorange": "reversed"},
        "height": 340
    })
    
    fig.update_layout(layout)
    return fig


def create_global_shap_chart(global_shap_df: pd.DataFrame) -> go.Figure:
    """
    Creates a horizontal Plotly bar chart showing Global Mean Absolute SHAP Feature Importance.
    """
    df_sorted = global_shap_df.sort_values(by="Mean_Absolute_SHAP", ascending=True)
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df_sorted["Feature"],
            x=df_sorted["Mean_Absolute_SHAP"],
            orientation="h",
            marker=dict(
                color=df_sorted["Mean_Absolute_SHAP"],
                colorscale=[[0.0, "#1F6FEB"], [1.0, "#58A6FF"]],
            ),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP| Value: %{x:.6f}<extra></extra>"
        )
    )
    
    layout = get_plotly_dark_layout()
    layout.update({
        "title": {"text": "Global Mean Absolute SHAP Feature Importance", "x": 0.0},
        "xaxis": {**layout["xaxis"], "title": "Mean |SHAP Value| (Impact on Model Output)"},
        "yaxis": {**layout["yaxis"], "title": "Feature"},
        "height": 320,
        "showlegend": False
    })
    
    fig.update_layout(layout)
    return fig


def create_local_shap_chart(local_shap_df: pd.DataFrame, latest_date: str, pred_state: int) -> go.Figure:
    """
    Creates a horizontal Plotly bar chart showing local SHAP contributions for the latest observation.
    """
    df_sorted = local_shap_df.sort_values(by="SHAP_Value", ascending=True)
    colors = [COLOR_ACCENT_GREEN if val >= 0 else COLOR_ACCENT_RED for val in df_sorted["SHAP_Value"]]
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df_sorted["Feature"],
            x=df_sorted["SHAP_Value"],
            orientation="h",
            marker_color=colors,
            customdata=df_sorted["Feature_Value"],
            hovertemplate="<b>%{y}</b><br>Feature Value: %{customdata:.6f}<br>SHAP Contribution: %{x:+.6f}<extra></extra>"
        )
    )
    
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=-0.5,
        y1=len(df_sorted) - 0.5,
        line=dict(color=COLOR_BORDER_LIGHT, width=1.5, dash="dash")
    )
    
    layout = get_plotly_dark_layout()
    layout.update({
        "title": {"text": f"Local SHAP Feature Contributions ({latest_date} — State {pred_state})", "x": 0.0},
        "xaxis": {**layout["xaxis"], "title": "SHAP Contribution Value (Push toward State)"},
        "yaxis": {**layout["yaxis"], "title": "Feature"},
        "height": 320,
        "showlegend": False
    })
    
    fig.update_layout(layout)
    return fig


def create_sentiment_distribution_chart(sentiment_counts: pd.Series) -> go.Figure:
    """
    Creates a Plotly Donut chart displaying FinBERT sentiment breakdown (Positive, Neutral, Negative).
    """
    labels = ["positive", "neutral", "negative"]
    counts = [sentiment_counts.get(l, 0) for l in labels]
    
    color_map = {
        "positive": COLOR_ACCENT_GREEN,
        "neutral": COLOR_TEXT_SECONDARY,
        "negative": COLOR_ACCENT_RED
    }
    colors = [color_map[l] for l in labels]
    
    fig = go.Figure(
        data=[
            go.Pie(
                labels=[l.capitalize() for l in labels],
                values=counts,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
            )
        ]
    )
    
    layout = get_plotly_dark_layout()
    layout.update({
        "title": {"text": "FinBERT Sentiment Distribution", "x": 0.0},
        "height": 300,
        "showlegend": True,
        "legend": {"orientation": "h", "y": -0.1, "x": 0.2, "font": {"color": COLOR_TEXT_SECONDARY}}
    })
    
    fig.update_layout(layout)
    return fig


def create_event_type_distribution_chart(event_stats_df: pd.DataFrame) -> go.Figure:
    """
    Creates a horizontal Plotly bar chart displaying article volume by event category.
    """
    df_sorted = event_stats_df.sort_values(by="Event_Count", ascending=True)
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df_sorted["event_type"],
            x=df_sorted["Event_Count"],
            orientation="h",
            marker=dict(
                color=df_sorted["Event_Count"],
                colorscale=[[0.0, "#1F6FEB"], [1.0, "#58A6FF"]]
            ),
            hovertemplate="<b>%{y}</b><br>Article Count: %{x}<extra></extra>"
        )
    )
    
    layout = get_plotly_dark_layout()
    layout.update({
        "title": {"text": "News Volume by Event Category", "x": 0.0},
        "xaxis": {**layout["xaxis"], "title": "Article Count"},
        "yaxis": {**layout["yaxis"], "title": "Event Category"},
        "height": 340,
        "showlegend": False
    })
    
    fig.update_layout(layout)
    return fig


def create_event_hmm_crosstab_heatmap(crosstab_df: pd.DataFrame) -> go.Figure:
    """
    Creates a Plotly Heatmap showing event frequency distribution across HMM states.
    """
    # Remove 'All' margin row/col for heatmap visualization
    clean_ct = crosstab_df[crosstab_df["event_type"] != "All"].copy()
    event_types = clean_ct["event_type"].tolist()
    
    state_cols = [c for c in clean_ct.columns if c not in ["event_type", "All"]]
    z_matrix = clean_ct[state_cols].values
    
    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=[f"HMM State {c}" for c in state_cols],
            y=event_types,
            text=[[str(val) for val in row] for row in z_matrix],
            texttemplate="%{text}",
            textfont={"size": 12, "color": COLOR_TEXT_PRIMARY},
            colorscale=[
                [0.0, "#0D1117"],
                [0.2, "#161B22"],
                [0.6, "#1F6FEB"],
                [1.0, "#58A6FF"]
            ],
            showscale=False,
            hovertemplate="<b>%{y}</b> @ <b>%{x}</b><br>Frequency: %{z} events<extra></extra>"
        )
    )
    
    layout = get_plotly_dark_layout()
    layout.update({
        "title": {"text": "Event Category × HMM Market State Crosstab", "x": 0.0},
        "xaxis": {**layout["xaxis"], "title": "Decoded HMM Market State"},
        "yaxis": {**layout["yaxis"], "title": "Event Category", "autorange": "reversed"},
        "height": 360
    })
    
    fig.update_layout(layout)
    return fig
