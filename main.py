from agent import app as agent_app
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
        "messages": [],
    }

    try:
        result = agent_app.invoke(initial_state)
    except Exception as e:
        print(f"[ERROR] Agent invocation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Agent processing failed. Please try again or contact support.",
        )

    # Extract the last message content if it exists, for a clean API response
    last_message = (
        result["messages"][-1].content if result.get("messages") else None
    )

    return {
        "classification": result.get("classification"),
        "reasoning": result.get("reasoning"),
        "tool_result": last_message,
    }


@api.get("/health")
async def health_check():
    return {"status": "ok"}