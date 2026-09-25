# Financial Market Regime & Event Intelligence Engine — Streamlit Frontend

Welcome to the frontend application for the **Financial Market Regime & Event Intelligence Engine**.

This application provides a modern dark analytics interface for exploring unsupervised Gaussian HMM market regimes, FinBERT news sentiment, rule-based event classification, and SHAP surrogate model explainability.

---

## Architecture Overview

The frontend is built using **Streamlit** and follow a modular component-driven architecture:

```text
app/
├── main.py                    # Main application entry point & router
├── components/                # Reusable UI component modules
│   ├── __init__.py
│   ├── sidebar.py            # Compact sidebar navigation
│   ├── cards.py              # Custom KPI cards, empty states, status badges
│   └── layout.py             # Page headers and section headers
├── pages/                     # Individual page renderers
│   ├── __init__.py
│   ├── overview.py           # Executive summary & snapshot page
│   ├── regimes.py            # HMM market regime trajectory & statistics
│   ├── events.py             # Financial news, FinBERT sentiment & forward returns
│   ├── explainability.py     # SHAP surrogate global & local explainability
│   └── model_health.py       # Pipeline telemetry & component health
├── utils/                     # Utility modules
│   ├── __init__.py
│   └── theme.py              # Centralized dark theme system & Plotly styling
└── README.md                  # Frontend documentation
```

---

## Design System & Styling (`app/utils/theme.py`)

The visual design overrides default Streamlit aesthetics to present a dark analytics platform:

- **Background Palette**: Near-black background (`#090C10`), dark panel surface (`#161B22`), and sidebar (`#0D1117`).
- **Typography & High Contrast**: Clean typography with muted secondary labels (`#8B949E`) and crisp white primary values (`#E6EDF3`).
- **Accent Colors**: Subtle blue (`#58A6FF`), green (`#3FB950`), amber (`#D29922`), and red (`#F85149`) status indicators.
- **Plotly Integration**: `get_plotly_dark_layout()` ensures all future interactive charts naturally conform to the dark theme without manual configuration.

---

## Page Structure

1. **Overview (`Overview`)**:
   - Executive market snapshot, active regime indicator, volatility signals, regime timeline placeholder, top SHAP drivers, and recent financial news stream.

2. **Market Regimes (`Market Regimes`)**:
   - HMM 4-state trajectory, empirical state statistics (mean return, vol, momentum, drawdown), transition probability matrix ($A$), and qualitative state taxonomy.

3. **News & Events (`News & Events`)**:
   - Financial news feed, FinBERT sentiment breakdown, rule-based event categories, and forward-return analysis ($t+1$ to $t+5$).

4. **Explainability (`Explainability`)**:
   - Random Forest surrogate fidelity metrics, global SHAP feature ranking across historical regimes, SHAP beeswarm plot, and local single-observation waterfall explanations.

5. **Model & Data Health (`Model & Data Health`)**:
   - Telemetry monitoring data coverage, observation counts, latest timestamps, pipeline execution status across Notebooks 01–08, and virtual environment details.

---

## Future Model Integration Blueprint

The UI components are decoupled from data ingestion and ML model execution. To connect offline research notebooks (Notebooks 01–08) or future live pipelines:

1. **Data Caching Layer**: Implement `@st.cache_data` or `@st.cache_resource` loaders in `app/utils/data_loader.py` to read processed data Parquet files or database tables.
2. **Replacing Placeholders**: Update renderer functions in `app/pages/*.py` to accept pandas DataFrames / Dict outputs instead of `render_empty_state()`.
3. **Interactive Charts**: Use `st.plotly_chart(fig, use_container_width=True)` with `get_plotly_dark_layout()` to render live regime overlays and SHAP bar charts.

---

## Running the Dashboard Locally

To launch the Streamlit frontend locally using the project's virtual environment:

```bash
.\.venv\Scripts\streamlit.exe run app/main.py
```
