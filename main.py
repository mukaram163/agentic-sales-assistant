from fastapi import FastAPI
from pydantic import BaseModel
from agent import app as agent_app

api = FastAPI()

class InquiryPayload(BaseModel):
    inquiry: str

@api.post("/webhook/inquiry")
async def handle_inquiry(payload: InquiryPayload):
    initial_state = {
        "lead_id": "",
        "inquiry": payload.inquiry,
        "classification": None,
        "reasoning": None,
        "messages": []
    }
    result = agent_app.invoke(initial_state)

    # extract the last message content if it exists, for a clean API response
    last_message = result["messages"][-1].content if result.get("messages") else None

    return {
        "classification": result.get("classification"),
        "reasoning": result.get("reasoning"),
        "tool_result": last_message
    }

@api.get("/health")
async def health_check():
    return {"status": "ok"}