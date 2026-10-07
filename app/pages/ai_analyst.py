"""
AI Analyst Page — Natural Language Financial Intelligence Interface
Financial Market Regime & Event Intelligence Engine
"""

import os
import requests
import streamlit as st
from app.components.layout import render_page_header, render_section_header
from app.components.cards import render_status_badge

# Default FastAPI Base URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def query_analyst_api(question: str, ticker: str = "^GSPC") -> dict:
    """
    Calls the FastAPI /api/analyst/query endpoint.
    Returns the JSON response dict or raises an Exception.
    """
    url = f"{API_BASE_URL.rstrip('/')}/api/analyst/query"
    payload = {"question": question, "ticker": ticker}
    
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def render():
    """Renders the AI Analyst Page."""
    
    # 1. Render Page Header
    render_page_header(
        title="AI Financial Intelligence Analyst",
        subtitle="Natural language query assistant grounded deterministically in persisted market regimes, news, and event analytics.",
        status_badge_text="FastAPI Backend Connected",
        status_type="success"
    )

    # 2. Section Header
    render_section_header(
        title="Ask the Financial Analyst",
        subtitle="Submit natural language questions to query live PostgreSQL market data, HMM regime predictions, and event analytics."
    )

    # 3. Example Questions & Input Form
    col_input, col_examples = st.columns([3, 2], gap="large")

    with col_examples:
        st.markdown("##### 💡 Example Questions")
        example_questions = [
            "What is the latest market regime?",
            "What is the latest market data?",
            "What recent financial news/events are available?",
            "How did Inflation perform?",
            "What was the market regime during Monetary Policy events?",
        ]
        
        selected_example = st.radio(
            "Select a sample question to try:",
            options=example_questions,
            index=0,
            key="example_radio"
        )

    with col_input:
        st.markdown("##### 📝 Enter Your Question")
        
        # User question input (defaulting to selected example)
        user_question = st.text_input(
            "Financial Question",
            value=selected_example,
            placeholder="e.g. What is the latest market regime?",
            key="user_question_input"
        )

        ticker_input = st.text_input(
            "Target Asset Ticker (Optional)",
            value="^GSPC",
            placeholder="^GSPC",
            key="ticker_input"
        )

        ask_button = st.button("🤖 Ask Analyst", type="primary", use_container_width=True)

    st.markdown("<hr style='margin: 1.5rem 0;' />", unsafe_allow_html=True)

    # 4. Handle Query Execution
    if ask_button or user_question:
        if not user_question.strip():
            st.warning("Please enter a question to query the analyst.")
            return

        with st.spinner("Analyzing question & querying PostgreSQL backend..."):
            try:
                result = query_analyst_api(user_question.strip(), ticker_input.strip() or "^GSPC")
                
                intent = result.get("intent", "unsupported")
                answer = result.get("answer", "")
                supporting_data = result.get("supporting_data", {})
                
                # Render Results
                render_section_header(
                    title="Analyst Response",
                    subtitle=f"Query: '{user_question}'"
                )

                # Intent Badge & Status
                badge_type = "warning" if intent == "unsupported" else "success"
                badge_html = render_status_badge(f"Intent: {intent}", badge_type=badge_type)
                
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
                        <h4 style="margin: 0; color: #E6EDF3;">Analyst Answer</h4>
                        {badge_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Answer Box
                if intent == "unsupported":
                    st.info(f"ℹ️ {answer}")
                else:
                    st.markdown(
                        f"""
                        <div style="background-color: #161B22; border: 1px solid #30363D; border-left: 4px solid #238636; border-radius: 6px; padding: 1.25rem; margin-bottom: 1.5rem;">
                            <div style="font-size: 1.05rem; color: #E6EDF3; line-height: 1.6;">
                                {answer}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Supporting Context & Data Expander
                with st.expander("🔍 View Supporting Data & Database Context", expanded=(intent != "unsupported")):
                    st.json(supporting_data)

            except requests.exceptions.ConnectionError:
                st.error("⚠️ Connection Error: Unable to reach the FastAPI backend server.")
                st.warning(
                    "Please ensure the FastAPI server is running on `http://localhost:8000` by executing:\n\n"
                    "```bash\n.\\.venv\\Scripts\\python.exe -m uvicorn backend.api.main:app --port 8000\n```"
                )
            except Exception as e:
                st.error(f"Error querying Analyst API: {str(e)}")
