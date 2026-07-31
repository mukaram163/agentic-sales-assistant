from langchain_core.tools import tool

from app.config import supabase


@tool
def create_qualified_lead(inquiry: str, reasoning: str) -> str:
    """Creates a new qualified lead record in the CRM database.
    Use this when an inquiry represents a genuine sales opportunity.
    """
    try:
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
    except Exception as e:
        print(f"[ERROR] Failed to create lead in Supabase: {e}")
        return "Lead creation failed due to a database error. This inquiry should be manually reviewed."
