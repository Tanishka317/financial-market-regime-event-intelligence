"""
LLM Prompt Templates & System Guidelines
Financial Market Regime & Event Intelligence Engine
"""

SYSTEM_PROMPT = """
You are an expert Financial Intelligence Analyst assisting a user with market regimes, quantitative asset statistics, financial news, and event analysis.

CRITICAL GROUNDING RULES:
1. Answer strictly using ONLY the provided Project Context (retrieved financial news/events) and Structured Database Statistics below.
2. Do NOT fabricate or extrapolate financial facts, prices, dates, or returns not present in the provided context.
3. If the provided context and statistics do not contain sufficient information to answer the question, state clearly: "Based on the available project data in PostgreSQL, I do not have enough information to answer this question."
4. Always distinguish observed historical market statistics (e.g., historical returns, volatility, past events) from predictions or subjective commentary.
5. Do NOT provide investment advice or ungrounded recommendations.
6. When referencing news or events, cite the headline, publisher, or event date from the provided context.
"""


def build_synthesis_prompt(
    question: str,
    deterministic_answer: str,
    supporting_data: dict,
    rag_sources: list
) -> str:
    """
    Constructs a structured prompt combining deterministic SQL results and RAG context.
    """
    prompt_parts = [f"USER QUESTION: {question}\n"]

    if deterministic_answer:
        prompt_parts.append(f"STRUCTURED DATABASE STATISTICAL ANALYSIS:\n{deterministic_answer}\n")

    if supporting_data:
        prompt_parts.append(f"SQL SUPPORTING DATA SCHEMATICS:\n{supporting_data}\n")

    if rag_sources:
        prompt_parts.append("RETRIEVED FINANCIAL NEWS & EVENT CONTEXT:")
        for idx, doc in enumerate(rag_sources, 1):
            text = doc.get("text", "")
            meta = doc.get("metadata", {})
            score = doc.get("score", "N/A")
            prompt_parts.append(
                f"--- Source [{idx}] (Relevance Score: {score}) ---\n"
                f"{text}\n"
                f"Metadata: Headline='{meta.get('headline', '')}', Date='{meta.get('published_at', '')}', EventType='{meta.get('event_type', '')}'\n"
            )
    else:
        prompt_parts.append("RETRIEVED FINANCIAL NEWS & EVENT CONTEXT: No relevant news records retrieved.\n")

    prompt_parts.append(
        "INSTRUCTIONS FOR YOUR RESPONSE:\n"
        "Synthesize a clear, concise, data-grounded response answering the user's question. "
        "Integrate quantitative SQL metrics with relevant news context where available. "
        "Adhere strictly to the grounding rules."
    )

    return "\n".join(prompt_parts)
