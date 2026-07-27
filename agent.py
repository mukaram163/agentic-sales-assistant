import os
from typing import Optional, TypedDict
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from supabase import create_client

load_dotenv()

# Initialize Supabase client
supabase = create_client(
    os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
)


# 1. Define the state — updated with messages field
class LeadState(TypedDict):
    lead_id: str
    inquiry: str
    classification: Optional[str]  # "qualified_lead", "support_question", "spam"
    reasoning: Optional[str]  # why the model classified it that way
    messages: list


# 2. Set up the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# 3. Define Tools
@tool
def create_qualified_lead(inquiry: str, reasoning: str) -> str:
    """Creates a new qualified lead record in the CRM database.
    Use this when an inquiry represents a genuine sales opportunity.
    """
    result = (
        supabase.table("leads")
        .insert(
            {
                "name": None,
                "contact": None,
                "source": "agent_tool_call",
                "inquiry": inquiry,
                "status": "qualifying",
                "conversation_history": [],
                "notes": reasoning,
            }
        )
        .execute()
    )
    new_id = result.data[0]["id"]
    return f"Lead created successfully with ID {new_id}, status set to 'qualifying'."


tools = [create_qualified_lead]
llm_with_tools = llm.bind_tools(tools)


# 4. Define Nodes
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

    # Basic parsing of the model's structured response
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


def agent_decide_action(state: LeadState) -> LeadState:
    prompt = f"""A lead was classified as: {state['classification']}
Reasoning: {state['reasoning']}
Original inquiry: "{state['inquiry']}"

If this is a qualified_lead, call the create_qualified_lead tool with the inquiry and reasoning.
Otherwise, do not call any tool.
"""
    response = llm_with_tools.invoke(prompt)
    state["messages"] = [response]
    return state


def handle_support_question(state: LeadState) -> LeadState:
    print(
        f"[SUPPORT] Would now escalate lead {state.get('lead_id', 'N/A')} to a human support queue."
    )
    return state


def handle_spam(state: LeadState) -> LeadState:
    print(
        f"[SPAM] Discarding inquiry from lead {state.get('lead_id', 'N/A')}."
    )
    return state


# 5. Routing functions
def route_after_classify(state: LeadState) -> str:
    classification = state.get("classification", "unknown")
    if classification == "qualified_lead":
        return "agent_decide_action"
    elif classification == "support_question":
        return "handle_support_question"
    else:
        return "handle_spam"


# 6. Build and compile graph
graph = StateGraph(LeadState)

graph.add_node("classify", classify_inquiry)
graph.add_node("agent_decide_action", agent_decide_action)
graph.add_node("tools", ToolNode(tools))
graph.add_node("handle_support_question", handle_support_question)
graph.add_node("handle_spam", handle_spam)

graph.set_entry_point("classify")

graph.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "agent_decide_action": "agent_decide_action",
        "handle_support_question": "handle_support_question",
        "handle_spam": "handle_spam",
    },
)

graph.add_edge("agent_decide_action", "tools")
graph.add_edge("tools", END)
graph.add_edge("handle_support_question", END)
graph.add_edge("handle_spam", END)

app = graph.compile()

# 7. Test execution block
if __name__ == "__main__":
    test_state = {
        "lead_id": "",
        "inquiry": "Hi, I'm interested in your services for my growing business, can we schedule a call?",
        "classification": None,
        "reasoning": None,
        "messages": [],
    }
    result = app.invoke(test_state)
    print("\nFinal State:")
    print(result)