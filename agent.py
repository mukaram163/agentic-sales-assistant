import os
from dotenv import load_dotenv
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

load_dotenv()

# 1. Define the state — this is what flows through every node in the graph
class LeadState(TypedDict):
    lead_id: str
    inquiry: str
    classification: Optional[str]   # "qualified_lead", "support_question", "spam"
    reasoning: Optional[str]        # why the model classified it that way

# 2. Set up the LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# 3. Classify node
def classify_inquiry(state: LeadState) -> LeadState:
    prompt = f"""You are triaging incoming business inquiries.

Classify this inquiry into exactly one category: qualified_lead, support_question, or spam.

Inquiry: "{state['inquiry']}"

Respond in this exact format:
Classification: <category>
Reasoning: <one sentence why>
"""
    response = llm.invoke(prompt)
    text = response.content

    # basic parsing of the model's structured response
    classification = "unknown"
    reasoning = text
    for line in text.split("\n"):
        if line.startswith("Classification:"):
            classification = line.split(":", 1)[1].strip()
        if line.startswith("Reasoning:"):
            reasoning = line.split(":", 1)[1].strip()

    state["classification"] = classification
    state["reasoning"] = reasoning
    return state

# 4. Handler nodes & Routing function
# New node: handle qualified leads — this is where CRM lookup/update will go later
def handle_qualified_lead(state: LeadState) -> LeadState:
    print(f"[QUALIFIED LEAD] Would now create/update lead {state['lead_id']} in Supabase and move to 'qualifying' status.")
    return state

# New node: handle support questions — this is where escalation logic will go later
def handle_support_question(state: LeadState) -> LeadState:
    print(f"[SUPPORT] Would now escalate lead {state['lead_id']} to a human support queue.")
    return state

# New node: handle spam — just log and discard
def handle_spam(state: LeadState) -> LeadState:
    print(f"[SPAM] Discarding inquiry from lead {state['lead_id']}.")
    return state

# Routing function: decides which node to go to next based on classification
def route_by_classification(state: LeadState) -> str:
    classification = state.get("classification", "unknown")
    if classification == "qualified_lead":
        return "handle_qualified_lead"
    elif classification == "support_question":
        return "handle_support_question"
    elif classification == "spam":
        return "handle_spam"
    else:
        return "handle_spam"  # fallback: treat unknown as spam for now, safer default

# 5. Build and compile the graph
graph = StateGraph(LeadState)
graph.add_node("classify", classify_inquiry)
graph.add_node("handle_qualified_lead", handle_qualified_lead)
graph.add_node("handle_support_question", handle_support_question)
graph.add_node("handle_spam", handle_spam)

graph.set_entry_point("classify")

graph.add_conditional_edges(
    "classify",
    route_by_classification,
    {
        "handle_qualified_lead": "handle_qualified_lead",
        "handle_support_question": "handle_support_question",
        "handle_spam": "handle_spam",
    }
)

graph.add_edge("handle_qualified_lead", END)
graph.add_edge("handle_support_question", END)
graph.add_edge("handle_spam", END)

app = graph.compile()

# 6. Test block
if __name__ == "__main__":
    test_state = {
        "lead_id": "test-123",
        "inquiry": "Hi, I'm interested in your services for my growing business, can we schedule a call?",
        "classification": None,
        "reasoning": None
    }
    result = app.invoke(test_state)
    print("\nFinal State:")
    print(result)