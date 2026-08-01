# Agentic Sales Assistant

An AI-powered sales assistant that automatically triages incoming business inquiries, qualifies leads, and books meetings — built with [LangGraph](https://github.com/langchain-ai/langgraph) and served via FastAPI.

[![CI](https://github.com/mukaram163/agentic-sales-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/mukaram163/agentic-sales-assistant/actions/workflows/ci.yml)

## What it does

The assistant receives an inquiry (via webhook — see the companion [n8n automation](https://github.com/mukaram163/n8n-lead-automation) that feeds it), then:

1. **Classifies** the inquiry as a qualified lead, a support question, or spam
2. **Acts autonomously** on qualified leads — creating a CRM record, checking calendar availability, and booking a meeting, calling tools in whatever order the situation requires
3. **Routes** support questions and spam to separate handlers instead of engaging the sales flow
4. **Returns a full audit trail** — every tool call and its result, plus a final plain-language summary — so a human can see exactly what the agent did and why

## Architecture

Built as a [LangGraph](https://github.com/langchain-ai/langgraph) state machine:

```
        ┌───────────┐
        │ classify  │
        └─────┬─────┘
              │
   ┌──────────┼──────────────┐
   ▼          ▼               ▼
qualified   support         spam
  lead      question       (discard)
   │          │
   ▼          ▼
agent_decide_action ◄──┐
   │                    │
   ▼                    │
 tools ─────────────────┘
(create_qualified_lead,
 check_calendar_availability,
 book_meeting)
```

The agent loops between deciding an action and executing tools until the task is complete or a max-step safety limit is hit, preventing runaway execution.

## Project structure

```
app/
├── config.py       # LLM and Supabase client setup
├── graph.py        # LangGraph state machine wiring
├── state.py        # Shared agent state definition
├── nodes/
│   ├── classify.py # Inquiry classification node
│   ├── agent.py    # Tool-calling decision loop
│   └── handlers.py # Support/spam handling
└── tools/
    ├── crm.py       # Lead creation (Supabase)
    └── calendar.py  # Availability check + booking (Supabase)
main.py              # FastAPI webhook entry point
tests/                # Unit tests (mocked external calls)
```

## Tech stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — agent orchestration and state management
- **[Groq](https://groq.com/)** (Llama 3.3 70B) — LLM inference
- **[Supabase](https://supabase.com/)** — CRM and calendar data
- **[FastAPI](https://fastapi.tiangolo.com/)** — webhook API
- **pytest** — unit testing with mocked external services
- **ruff** — linting
- **GitHub Actions** — CI on every push and PR

## Getting started

### Prerequisites

- Python 3.11+
- A Groq API key
- A Supabase project with `leads` and `availability` tables

### Setup

```bash
git clone https://github.com/mukaram163/agentic-sales-assistant.git
cd agentic-sales-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

### Run locally

```bash
uvicorn main:api --reload
```

Send a test inquiry:

```bash
curl -X POST "http://127.0.0.1:8000/webhook/inquiry" \
     -H "Content-Type: application/json" \
     -d '{"inquiry": "Hi, I am interested in your services, can we schedule a call?"}'
```

### Run tests

```bash
pytest
```

## Related

- [n8n-lead-automation](https://github.com/mukaram163/n8n-lead-automation) — the n8n workflow that feeds inquiries into this service

## License

MIT
