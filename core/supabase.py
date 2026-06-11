# core/supabase.py
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str | None = (
    os.environ.get("SUPABASE_URL")
    or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
)
key: str | None = (
    os.environ.get("SUPABASE_KEY")
    or os.environ.get("SUPABASE_ANON_KEY")
    or os.environ.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
)

if not url or not key:
    raise ValueError(
        "Supabase keys are missing from .env file. Set SUPABASE_URL/SUPABASE_KEY "
        "or NEXT_PUBLIC_SUPABASE_URL/NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY."
    )

# Export this variable so other files can use it
db: Client = create_client(url, key)