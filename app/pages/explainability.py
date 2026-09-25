"""
Explainability & SHAP Surrogate Dashboard Page — Connected to Notebook 08 Pipeline
Financial Market Regime & Event Intelligence Engine
"""

import streamlit as st
import pandas as pd
import numpy as np
from app.components.layout import render_page_header, render_section_header
from app.components.cards import render_kpi_card, render_empty_state
from app.utils.explainability import compute_shap_explainability
from app.utils.shared_context import get_shared_dashboard_context
from app.utils.charts import (
    create_global_shap_chart,
    create_local_shap_chart
)


def render():
    """Renders the Explainability Page connected to real Random Forest surrogate & SHAP values."""
    
    # 1. Fetch shared dashboard context & compute SHAP explainability results
    try:
        with st.spinner("Computing SHAP surrogate feature attributions..."):
            ctx = get_shared_dashboard_context(period="5y")
            res = compute_shap_explainability(period="5y")
            
            test_acc = res["test_accuracy"]
            macro_f1_act = res["macro_f1_active"]
            macro_f1_all = res["macro_f1_all"]
            test_n = res["test_samples"]
            active_states = res["active_test_states"]
            global_df = res["global_shap_df"]
            latest_date = res["latest_date"]
            actual_state = res["actual_hmm_state"]
            pred_state = res["surrogate_pred_state"]
            is_match = res["is_match"]
            latest_feats = res["latest_features"]
            local_df = res["local_shap_df"]
            
            curr_state = ctx["current_state"]
            curr_label = ctx["current_regime_label"]
            
            status_text = f"Surrogate Fidelity: {test_acc*100:.1f}%"
            status_type = "success"
            data_error = None
    except Exception as e:
        status_text = "SHAP Explainer Unavailable"
        status_type = "warning"
        data_error = str(e)

    # 2. Render Page Header
    render_page_header(
        title="Regime Explainability (SHAP Surrogate)",
        subtitle="Why did the model identify this market regime? Game-theoretic SHAP attributions explaining Random Forest surrogate decision boundaries.",
        status_badge_text=status_text,
        status_type=status_type
    )

    if data_error:
        st.error(f"SHAP Explainability Error: {data_error}")
        render_empty_state(
            title="Unable to Compute SHAP Attributions",
            subtitle="Check HMM model and Random Forest surrogate execution status.",
            icon="⚠️"
        )
        return

    # 3. PIPELINE CONNECTION & ACTIVE REGIME BANNER
    st.markdown(
        f"""
        <div class="fmie-card" style="border-left: 4px solid #58A6FF; margin-bottom: 1.2rem; padding: 0.9rem 1.2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem;">
                <div>
                    <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: #58A6FF; font-weight: 700;">
                        Active Market Regime & Surrogate Connection
                    </div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #E6EDF3; margin-top: 0.15rem;">
                        Current HMM State {curr_state} ({curr_label}) &nbsp; ──▶ &nbsp; Random Forest Surrogate &nbsp; ──▶ &nbsp; SHAP Attribution Breakdown
                    </div>
                </div>
                <div style="font-size: 0.78rem; color: #8B949E; background: #0D1117; padding: 0.35rem 0.7rem; border-radius: 6px; border: 1px solid #30363D;">
                    Session Date: <strong>{latest_date}</strong>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #8B949E; margin-top: 0.5rem; line-height: 1.4;">
                *Methodological Rule*: The Random Forest surrogate approximates the unsupervised HMM state decodes. SHAP attributions explain 
                <em>why the surrogate classifier assigned this session to State {pred_state}</em> based on quantitative market features.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4. SECTION 1: SURROGATE FIDELITY (KPI Cards)
    render_section_header(
        title="1. Out-of-Sample Surrogate Fidelity",
        subtitle="Evaluating how accurately the Random Forest surrogate reproduces HMM decoded state assignments (Chronological 80/20 Test Split)"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            title="Surrogate Test Accuracy",
            value=f"{test_acc*100:.2f}%",
            subtitle="Out-of-sample state fidelity",
            badge_text="Test Fidelity",
            badge_type="success"
        )
    with col2:
        render_kpi_card(
            title="Macro F1 (Active Test States)",
            value=f"{macro_f1_act:.4f}",
            subtitle=f"F1 across active states {active_states}",
            badge_text="Active Macro F1",
            badge_type="research"
        )
    with col3:
        render_kpi_card(
            title="Test Observations (N)",
            value=f"{test_n} Sessions",
            subtitle="Chronological 20% test window",
            badge_text="Test Set N",
            badge_type="research"
        )
    with col4:
        active_states_str = ", ".join([f"State {s}" for s in active_states])
        render_kpi_card(
            title="Active Test Regimes",
            value=active_states_str,
            subtitle=f"All 4 states evaluated in global SHAP",
            badge_text="Active States",
            badge_type="research"
        )

    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #8B949E; margin-top: 0.25rem;">
            *Note*: <strong>Surrogate Fidelity</strong> evaluates model approximation quality (how accurately the Random Forest mimics HMM state decodes). 
            It is strictly distinct from future market price prediction accuracy.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. SECTION 2: GLOBAL REGIME DRIVERS (Horizontal Bar Chart & Table)
    render_section_header(
        title="2. Global Regime Drivers Across Historical Regimes",
        subtitle="Mean absolute SHAP value ranking evaluated across 5-year multi-regime dataset (N=1,235), ensuring all 4 HMM states are represented"
    )

    g_chart_col, g_table_col = st.columns([1.2, 0.8])
    with g_chart_col:
        fig_global = create_global_shap_chart(global_df)
        st.plotly_chart(fig_global, use_container_width=True)

    with g_table_col:
        st.markdown(
            """
            <div style="font-weight: 600; color: #E6EDF3; font-size: 0.95rem; margin-bottom: 0.5rem;">
                Global SHAP Ranking Table
            </div>
            """,
            unsafe_allow_html=True
        )
        global_display_df = pd.DataFrame({
            "Rank": [f"Rank {i+1}" for i in range(len(global_df))],
            "Feature": global_df["Feature"],
            "Mean |SHAP| Value": [f"{v:.6f}" for v in global_df["Mean_Absolute_SHAP"]]
        })
        st.dataframe(global_display_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 6. SECTION 3: WHY THIS REGIME? (Local Observation Explanation)
    render_section_header(
        title=f"3. Why This Regime? Local Explanation for {latest_date}",
        subtitle=f"Single-session SHAP feature contributions explaining why the surrogate assigned the session to State {pred_state}"
    )

    match_badge_type = "success" if is_match else "warning"
    match_str = "MATCH (Surrogate = HMM)" if is_match else "MISMATCH"

    loc_kpi_col1, loc_kpi_col2, loc_kpi_col3, loc_kpi_col4 = st.columns(4)
    with loc_kpi_col1:
        render_kpi_card(
            title="Observation Date",
            value=latest_date,
            subtitle="Latest session in market dataset",
            badge_text="Latest Date",
            badge_type="research"
        )
    with loc_kpi_col2:
        render_kpi_card(
            title="Actual HMM State",
            value=f"State {actual_state} ({curr_label})",
            subtitle="Unsupervised Gaussian HMM decode",
            badge_text="HMM Target",
            badge_type="success"
        )
    with loc_kpi_col3:
        render_kpi_card(
            title="Surrogate Predicted",
            value=f"State {pred_state}",
            subtitle="Random Forest surrogate prediction",
            badge_text="Surrogate Pred",
            badge_type="research"
        )
    with loc_kpi_col4:
        render_kpi_card(
            title="Prediction Match",
            value=match_str,
            subtitle="Surrogate fidelity check",
            badge_text="Match Status",
            badge_type=match_badge_type
        )

    fig_local = create_local_shap_chart(local_df, latest_date, pred_state)
    st.plotly_chart(fig_local, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 7. SECTION 4: FEATURE SNAPSHOT (Latest Session Feature Values)
    render_section_header(
        title=f"4. Latest Feature Snapshot ({latest_date})",
        subtitle="Raw quantitative feature inputs feeding the regime decision engine for the latest session"
    )

    feat_cols = st.columns(6)
    with feat_cols[0]:
        v = latest_feats["Daily_Return"] * 100
        render_kpi_card(title="Daily Return", value=f"{v:+.2f}%", subtitle="1-day return", badge_text="Return", badge_type="research")
    with feat_cols[1]:
        v = latest_feats["Rolling_Volatility_20"] * 100
        render_kpi_card(title="20D Volatility", value=f"{v:.2f}%", subtitle="20D std dev", badge_text="Vol", badge_type="research")
    with feat_cols[2]:
        v = latest_feats["Momentum_20"] * 100
        render_kpi_card(title="20D Momentum", value=f"{v:+.2f}%", subtitle="20D price change", badge_text="Mom", badge_type="research")
    with feat_cols[3]:
        v = latest_feats["Drawdown"] * 100
        render_kpi_card(title="Drawdown Depth", value=f"{v:.2f}%", subtitle="Peak-to-trough", badge_text="Drawdown", badge_type="warning")
    with feat_cols[4]:
        v = latest_feats["SP500_TLT_Corr_20"]
        render_kpi_card(title="SP500 ↔ TLT Corr", value=f"{v:+.3f}", subtitle="Treasury bond corr", badge_text="Corr", badge_type="research")
    with feat_cols[5]:
        v = latest_feats["SP500_GLD_Corr_20"]
        render_kpi_card(title="SP500 ↔ GLD Corr", value=f"{v:+.3f}", subtitle="Gold commodity corr", badge_text="Corr", badge_type="research")

    st.markdown("<br>", unsafe_allow_html=True)

    # 8. SECTION 5: METHODOLOGY & LIMITATIONS
    render_section_header(
        title="5. SHAP Methodology & Model Disclaimers",
        subtitle="Conceptual boundaries and interpretation rules for explainable market regime surrogates"
    )

    with st.expander("📖 View Detailed SHAP Methodology & Conceptual Limitations", expanded=True):
        st.markdown(
            """
            1. **Unsupervised HMM Mechanics**: The Gaussian HMM is an *unsupervised sequential latent-variable model* governed by transition matrices and continuous multivariate emission distributions.
            2. **Supervised Surrogate Model**: The `RandomForestClassifier` (`n_estimators=300`, `random_state=42`) is a *surrogate model* trained to approximate the HMM's decoded state assignments ($s_t$) from the 6 quantitative features ($X_t$).
            3. **Game-Theoretic SHAP Attributions**: SHAP `TreeExplainer` calculates exact marginal Shapley feature contributions explaining the decision boundaries of the Random Forest surrogate.
            4. **Indirect HMM Explanation**: SHAP explains the decision logic of the surrogate tree ensemble, **NOT** the internal matrix transition or emission likelihood mechanisms of the original HMM directly.
            5. **Multicollinearity & Feature Dependence**: Correlated financial features (e.g., Drawdown, Volatility, and Momentum) distribute SHAP attribution values across co-linear variables.
            6. **Observational Scope**: This analysis is strictly historical and observational. It does not constitute trading signals or financial advice.
            7. **Surrogate Fidelity Constraint**: The interpretation of SHAP feature importance relies on maintaining high out-of-sample surrogate fidelity.
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    render()

