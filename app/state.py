from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class LeadState(TypedDict):
    lead_id: str
    inquiry: str
    classification: str | None
    reasoning: str | None
    messages: Annotated[list, add_messages]
    steps: int
