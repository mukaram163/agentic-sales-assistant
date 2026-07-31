from langchain_core.tools import tool

from app.config import supabase


@tool
def check_calendar_availability() -> str:
    """Checks available meeting slots that are not yet booked.
    Use this when a qualified lead wants to schedule a call.
    Returns a list of available time slots.
    """
    try:
        result = (
            supabase.table("availability")
            .select("*")
            .eq("is_booked", False)
            .order("slot_time")
            .limit(3)
            .execute()
        )
        if not result.data:
            return "No available slots found. Escalate to a human to manually schedule."
        slots = [f"{row['id']}: {row['slot_time']}" for row in result.data]
        return "Available slots:\n" + "\n".join(slots)
    except Exception as e:
        print(f"[ERROR] Failed to check availability: {e}")
        return "Could not check calendar availability due to a system error."


@tool
def book_meeting(slot_id: str, lead_id: str) -> str:
    """Books a specific meeting slot for a lead.
    Use this after check_calendar_availability, once a specific slot_id has been chosen.
    """
    try:
        result = (
            supabase.table("availability")
            .update({"is_booked": True, "lead_id": lead_id})
            .eq("id", slot_id)
            .eq("is_booked", False)
            .execute()
        )
        if not result.data:
            return "That slot is no longer available. Please check availability again."
        return f"Meeting successfully booked for slot {slot_id}."
    except Exception as e:
        print(f"[ERROR] Failed to book meeting: {e}")
        return "Could not book the meeting due to a system error."
