import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_KEY")
)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)