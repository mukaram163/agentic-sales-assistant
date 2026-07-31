from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.nodes.agent import agent_decide_action, should_continue
from app.nodes.classify import classify_inquiry
from app.nodes.handlers import handle_spam, handle_support_question
from app.state import LeadState
from app.tools import tools


def route_after_classify(state: LeadState) -> str:
    classification = state.get("classification", "unknown")
    if classification == "qualified_lead":
        return "agent_decide_action"
    elif classification == "support_question":
        return "handle_support_question"
    else:
        return "handle_spam"


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

graph.add_conditional_edges(
    "agent_decide_action",
    should_continue,
    {"tools": "tools", "end": END},
)

graph.add_edge("tools", "agent_decide_action")
graph.add_edge("handle_support_question", END)
graph.add_edge("handle_spam", END)

app = graph.compile()