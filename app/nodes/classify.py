from app.config import llm
from app.state import LeadState


def classify_inquiry(state: LeadState) -> LeadState:
    prompt = f"""You are triaging incoming business inquiries.

Classify this inquiry into exactly one category: qualified_lead, support_question, or spam.

Inquiry: "{state['inquiry']}"

Respond in this exact format:
Classification: <category>
Reasoning: <one sentence why>
"""
    try:
        response = llm.invoke(prompt)
        text = response.content
    except Exception as e:
        print(f"[ERROR] LLM call failed during classification: {e}")
        state["classification"] = "unknown"
        state["reasoning"] = "Classification failed due to an LLM API error."
        return state

    classification = "unknown"
    reasoning = text
    for line in text.split("\n"):
        if line.startswith("Classification:"):
            classification = line.split(":", 1)[1].strip()
        if line.startswith("Reasoning:"):
            reasoning = line.split(":", 1)[1].strip()

    if classification not in ("qualified_lead", "support_question", "spam"):
        print(f"[WARNING] Unexpected classification value: '{classification}'. Defaulting to safe fallback.")

    state["classification"] = classification
    state["reasoning"] = reasoning
    return state
