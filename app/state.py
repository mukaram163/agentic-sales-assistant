from typing import Annotated, Optional, TypedDict
from langgraph.graph.message import add_messages


class LeadState(TypedDict):
    lead_id: str
    inquiry: str
    classification: Optional[str]
    reasoning: Optional[str]
    messages: Annotated[list, add_messages]
    steps: int