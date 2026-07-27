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

# 3. Build one node: classify the inquiry
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

# 4. Build the graph with just this one node for now
graph = StateGraph(LeadState)
graph.add_node("classify", classify_inquiry)
graph.set_entry_point("classify")
graph.add_edge("classify", END)

app = graph.compile()

# 5. Test it
if __name__ == "__main__":
    test_state = {
        "lead_id": "test-123",
        "inquiry": "Hi, I'm interested in your services for my growing business, can we schedule a call?",
        "classification": None,
        "reasoning": None
    }
    result = app.invoke(test_state)
    print(result)