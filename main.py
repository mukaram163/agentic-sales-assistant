from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import AIMessage, ToolMessage
from app.graph import app as agent_app

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
        "steps": 0,
    }

    try:
        result = agent_app.invoke(initial_state)
    except Exception as e:
        print(f"[ERROR] Agent invocation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Agent processing failed. Please try again or contact support.",
        )

    # Build a clean trail of actions taken, skipping system/human messages
    actions_taken = []
    for m in result.get("messages", []):
        if isinstance(m, ToolMessage):
            actions_taken.append({"tool": m.name, "result": m.content})

    # Extract the last non-empty AI summary message
    final_summary = None
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage) and m.content:
            final_summary = m.content
            break

    return {
        "classification": result.get("classification"),
        "reasoning": result.get("reasoning"),
        "actions_taken": actions_taken,
        "final_summary": final_summary,
    }


@api.get("/health")
async def health_check():
    return {"status": "ok"}