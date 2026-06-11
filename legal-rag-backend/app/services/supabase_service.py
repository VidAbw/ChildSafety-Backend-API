from app.config import config
from app.schemas.legal_schema import LegalSection
from typing import List, Dict

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

supabase = None
if create_client and config.SUPABASE_URL and config.SUPABASE_ANON_KEY:
    try:
        supabase = create_client(config.SUPABASE_URL, config.SUPABASE_ANON_KEY)
    except Exception:
        supabase = None


def get_legal_sections() -> List[LegalSection]:
    if not supabase:
        return []
    response = supabase.table('legal_sections').select('*').execute()
    return [LegalSection(**item) for item in response.data or []]


def get_reporting_contacts() -> List[Dict]:
    if not supabase:
        return []
    response = supabase.table('reporting_contacts').select('*').execute()
    return response.data or []


def save_feedback(abuse_category: str, retrieved_section: str, rating: str, comment: str = ""):
    if not supabase:
        return
    supabase.table('feedback').insert({
        'abuse_category': abuse_category,
        'retrieved_section': retrieved_section,
        'rating': rating,
        'comment': comment
    }).execute()