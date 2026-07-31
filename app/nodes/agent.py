from langchain_core.messages import HumanMessage, SystemMessage

from app.config import llm
from app.state import LeadState
from app.tools import tools

llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are an AI sales assistant. You have access to these tools:
- create_qualified_lead: creates a new lead record in the CRM. Call this first for any qualified lead.
- check_calendar_availability: checks open meeting slots. Call this if the lead wants to schedule a call.
- book_meeting: books a specific slot_id for a lead_id. Call this once you have both a slot_id (from check_calendar_availability) and a lead_id (from create_qualified_lead), if the inquiry asked for a call or meeting.

Work through these steps one at a time based on what the inquiry needs. Once you have taken all necessary actions, respond with a final plain-text summary and do not call any more tools.
"""

MAX_STEPS = 5


def agent_decide_action(state: LeadState) -> dict:
    if state["steps"] == 0:
        conversation = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Classification: {state['classification']}\nReasoning: {state['reasoning']}\nInquiry: \"{state['inquiry']}\""
            ),
        ]
        response = llm_with_tools.invoke(conversation)
        new_messages = conversation + [response]
    else:
        response = llm_with_tools.invoke(state["messages"])
        new_messages = [response]

    return {"messages": new_messages, "steps": state["steps"] + 1}


def should_continue(state: LeadState) -> str:
    last_message = state["messages"][-1]
    if state["steps"] >= MAX_STEPS:
        print(f"[WARNING] Hit max steps ({MAX_STEPS}), stopping loop to prevent runaway execution.")
        return "end"
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"