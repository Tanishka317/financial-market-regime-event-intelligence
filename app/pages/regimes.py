"""
Market Regimes Dashboard Page — Connected to Gaussian HMM Pipeline
Financial Market Regime & Event Intelligence Engine
"""

import streamlit as st
import pandas as pd
import numpy as np
from app.components.layout import render_page_header, render_section_header
from app.components.cards import render_kpi_card, render_empty_state
from app.components.sidebar import navigate_to
from app.utils.regime_model import fit_and_decode_hmm
from app.utils.shared_context import get_shared_dashboard_context
from app.utils.charts import (
    create_hmm_regime_timeline_chart,
    create_transition_matrix_heatmap
)


def render():
    """Renders the Market Regimes Page connected to real Gaussian HMM decodes."""
    
    # 1. Fetch & decode HMM model results via shared context & regime_model
    try:
        with st.spinner("Decoding HMM market regimes..."):
            ctx = get_shared_dashboard_context(period="5y")
            res = fit_and_decode_hmm(period="5y")
            market_df = res["market_df"]
            transmat = res["transmat"]
            stats_df = res["state_stats_df"]
            duration_df = res["duration_df"]
            
            latest_state = ctx["current_state"]
            latest_date = ctx["latest_date"]
            latest_label = ctx["current_regime_label"]
            latest_desc = ctx["current_regime_desc"]
            consec_days = ctx["consecutive_sessions"]
            expected_dur = ctx["expected_duration"]
            empirical_dur = ctx["empirical_duration"]
            p_ii = ctx["p_ii"]
            next_state = ctx["next_likely_state"]
            next_prob = ctx["next_likely_prob"]
            log_lik = res["log_likelihood"]
            
            status_text = f"Active: State {latest_state} ({latest_label})"
            status_type = "success"
            data_error = None
    except Exception as e:
        market_df = None
        status_text = "HMM Model Unavailable"
        status_type = "warning"
        data_error = str(e)

    # 2. Render Page Header
    render_page_header(
        title="Market Regime Detection & Trajectory",
        subtitle="Unsupervised 4-state Gaussian Hidden Markov Model (HMM) regime decoding across multi-asset quantitative features.",
        status_badge_text=status_text,
        status_type=status_type
    )

    if data_error:
        st.error(f"Gaussian HMM Pipeline Error: {data_error}")
        render_empty_state(
            title="Unable to Fit HMM Model",
            subtitle="Ensure market dataset contains sufficient observations.",
            icon="⚠️"
        )
        return

    # 3. CURRENT REGIME SNAPSHOT: Top KPI Cards Row & Narrative Context
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card(
            title="Current Regime",
            value=f"State {latest_state}",
            subtitle=latest_label,
            badge_text="Active State",
            badge_type="success"
        )
    with col2:
        render_kpi_card(
            title="Current Streak",
            value=f"{consec_days} Sessions",
            subtitle=f"Since {latest_date}",
            badge_text="Persistence",
            badge_type="research"
        )
    with col3:
        render_kpi_card(
            title="Expected Duration",
            value=f"{expected_dur:.1f} Days",
            subtitle=f"Theoretical E[D] = 1/(1-P_ii)",
            badge_text="Model Duration",
            badge_type="research"
        )
    with col4:
        render_kpi_card(
            title="Empirical Duration",
            value=f"{empirical_dur:.1f} Days",
            subtitle="Observed block length",
            badge_text="Observed Avg",
            badge_type="research"
        )
    with col5:
        render_kpi_card(
            title="Transition Probabilities",
            value=f"P_{latest_state}{latest_state}: {p_ii*100:.1f}%",
            subtitle=f"Next likely: State {next_state} ({next_prob*100:.1f}%)",
            badge_text="Transmat",
            badge_type="research"
        )

    st.markdown(
        f"""
        <div class="fmie-card" style="border-left: 4px solid #58A6FF; margin-top: 0.5rem; margin-bottom: 1.2rem;">
            <div style="font-weight: 600; color: #E6EDF3; font-size: 0.95rem; margin-bottom: 0.3rem;">
                Current Regime Intelligence (As of {latest_date}):
            </div>
            <div style="font-size: 0.85rem; color: #8B949E; line-height: 1.5;">
                The market is currently in <strong>State {latest_state} ({latest_label})</strong>. {latest_desc}
                This regime exhibits a high self-persistence probability of <strong>{p_ii*100:.1f}%</strong> per trading session, 
                with a theoretical expected duration of <strong>{expected_dur:.1f} sessions</strong> and an empirical average duration of <strong>{empirical_dur:.1f} sessions</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4. REGIME INTELLIGENCE SECTION: Cross-Module Links
    render_section_header(
        title="REGIME INTELLIGENCE & CROSS-MODULE NAVIGATION",
        subtitle="Connect latent HMM market regimes to financial news streams and feature attribution explainability."
    )

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown(
            f"""
            <div class="fmie-card" style="margin-bottom: 0.5rem;">
                <div style="font-weight: 600; color: #E6EDF3; font-size: 0.9rem; margin-bottom: 0.3rem;">
                    📰 News & Event Associations
                </div>
                <div style="font-size: 0.8rem; color: #8B949E; margin-bottom: 0.75rem;">
                    Inspect financial news headlines, FinBERT sentiment distributions, and event categories associated with State {latest_state} ({latest_label}).
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button(f"Inspect Historical Events in State {latest_state} →", use_container_width=True, key="btn_regime_to_events"):
            navigate_to("News & Events")

    with r_col2:
        st.markdown(
            f"""
            <div class="fmie-card" style="margin-bottom: 0.5rem;">
                <div style="font-weight: 600; color: #E6EDF3; font-size: 0.9rem; margin-bottom: 0.3rem;">
                    🔍 Feature Drivers & SHAP Attributions
                </div>
                <div style="font-size: 0.8rem; color: #8B949E; margin-bottom: 0.75rem;">
                    Analyze game-theoretic SHAP feature attributions explaining why the surrogate model assigned the market to State {latest_state}.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button(f"Analyze Feature Drivers for State {latest_state} →", use_container_width=True, key="btn_regime_to_shap"):
            navigate_to("Explainability")

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. REGIME TIMELINE: Interactive Plotly Timeline Chart
    render_section_header(
        title="1. Historical Regime Trajectory Timeline",
        subtitle=f"Decoded Gaussian HMM market regimes overlaid on S&P 500 daily price trajectory ({len(market_df):,} sessions)"
    )
    fig_timeline = create_hmm_regime_timeline_chart(market_df)
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 6. REGIME CHARACTERISTICS: Qualitative Cards Derived from Actual Statistics
    render_section_header(
        title="2. Regime Characteristics & Empirical Taxonomy",
        subtitle="Feature profiles derived directly from fitted model emissions (Daily Return, Volatility, Momentum, Drawdown)"
    )

    char_cols = st.columns(4)
    for state_idx in range(4):
        row = stats_df.loc[stats_df["state"] == state_idx].iloc[0]
        state_cnt = row["count"]
        state_pct = row["percentage"]
        lbl = row["label"]
        desc = row["description"]
        m_ret = row["mean_daily_return"] * 100
        m_vol = row["mean_volatility_20"] * 100
        m_mom = row["mean_momentum_20"] * 100
        m_dd = row["mean_drawdown"] * 100

        state_color_map = {0: "#3FB950", 1: "#F85149", 2: "#D29922", 3: "#58A6FF"}
        card_color = state_color_map.get(state_idx, "#E6EDF3")
        is_active = (state_idx == latest_state)
        border_style = f"border: 2px solid {card_color};" if is_active else "border: 1px solid #30363D;"

        with char_cols[state_idx]:
            active_badge = f'<span style="background-color: {card_color}; color: #0D1117; font-size: 0.65rem; font-weight: 700; padding: 0.1rem 0.4rem; border-radius: 8px; float: right;">ACTIVE</span>' if is_active else ''
            st.markdown(
                f"""
                <div class="fmie-card" style="{border_style} min-height: 220px;">
                    {active_badge}
                    <div style="font-weight: 700; color: {card_color}; margin-bottom: 0.2rem;">State {state_idx}</div>
                    <div style="font-size: 0.9rem; font-weight: 600; color: #E6EDF3; margin-bottom: 0.4rem;">{lbl}</div>
                    <div style="font-size: 0.76rem; color: #8B949E; margin-bottom: 0.6rem; min-height: 48px;">{desc}</div>
                    <hr style="margin: 0.4rem 0 !important; border-color: #21262D !important;" />
                    <div style="font-size: 0.74rem; color: #6E7681;">
                        <div>• Count: <strong>{state_cnt}</strong> ({state_pct:.1f}%)</div>
                        <div>• Mean Vol: <strong>{m_vol:.2f}%</strong></div>
                        <div>• Mean Mom: <strong>{m_mom:+.2f}%</strong></div>
                        <div>• Mean DD: <strong>{m_dd:.2f}%</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # 7. TRANSITION MATRIX + DURATION: Heatmap & State Persistence Table
    render_section_header(
        title="3. Transition Matrix & State Persistence",
        subtitle="Inter-regime transition probability matrix and expected vs. empirical state duration"
    )

    matrix_col, duration_col = st.columns([1.1, 0.9])

    with matrix_col:
        fig_transmat = create_transition_matrix_heatmap(transmat)
        st.plotly_chart(fig_transmat, use_container_width=True)
        st.caption(
            "The transition matrix represents the probability of moving from one latent market regime to another between consecutive observations. "
            "Diagonal entries (P_ii) represent state persistence (probability of remaining in the same state)."
        )

    with duration_col:
        st.markdown(
            """
            <div style="font-weight: 600; color: #E6EDF3; font-size: 0.95rem; margin-bottom: 0.5rem;">
                State Duration Telemetry (Trading Sessions)
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display duration comparison table
        dur_display_df = pd.DataFrame({
            "Regime State": [f"State {row['state']} ({stats_df.loc[stats_df['state']==row['state'], 'label'].values[0]})" for _, row in duration_df.iterrows()],
            "Persistence (P_ii)": [f"{row['p_ii']:.4f}" for _, row in duration_df.iterrows()],
            "Model Expected": [f"{row['model_expected_duration']:.1f} days" for _, row in duration_df.iterrows()],
            "Empirical Observed": [f"{row['empirical_avg_duration']:.1f} days" for _, row in duration_df.iterrows()]
        })
        st.dataframe(dur_display_df, use_container_width=True, hide_index=True)
        
        st.caption(
            "Model-implied expected duration is calculated via E[D_i] = 1 / (1 - P_ii). "
            "Empirical average duration measures the actual observed length of consecutive regime blocks in the decoded time series."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 8. REGIME STATISTICS: Full Empirical Table
    render_section_header(
        title="4. Full Empirical Regime Statistics",
        subtitle="Comprehensive numerical summary of feature means across all 4 Gaussian HMM market states"
    )

    formatted_stats_df = pd.DataFrame({
        "State": [f"State {row['state']}" for _, row in stats_df.iterrows()],
        "Regime Label": stats_df["label"],
        "Count (N)": stats_df["count"],
        "Pct (%)": [f"{v:.1f}%" for v in stats_df["percentage"]],
        "Mean Return": [f"{v*100:+.2f}%" for v in stats_df["mean_daily_return"]],
        "Mean 20D Vol": [f"{v*100:.2f}%" for v in stats_df["mean_volatility_20"]],
        "Mean 20D Mom": [f"{v*100:+.2f}%" for v in stats_df["mean_momentum_20"]],
        "Mean Drawdown": [f"{v*100:.2f}%" for v in stats_df["mean_drawdown"]],
        "SP500 ↔ TLT Corr": [f"{v:+.3f}" for v in stats_df["mean_tlt_corr"]],
        "SP500 ↔ GLD Corr": [f"{v:+.3f}" for v in stats_df["mean_gld_corr"]]
    })

    st.dataframe(formatted_stats_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render()

