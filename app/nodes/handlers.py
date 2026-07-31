from app.state import LeadState


def handle_support_question(state: LeadState) -> LeadState:
    print(f"[SUPPORT] Would now escalate lead {state.get('lead_id', 'N/A')} to a human support queue.")
    return state


def handle_spam(state: LeadState) -> LeadState:
    print(f"[SPAM] Discarding inquiry from lead {state.get('lead_id', 'N/A')}.")
    return state