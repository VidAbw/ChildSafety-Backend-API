from supabase import create_client, Client
from app.config import config
from app.schemas.legal_schema import LegalSection
from typing import List, Dict

supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_ANON_KEY)

def get_legal_sections() -> List[LegalSection]:
    response = supabase.table('legal_sections').select('*').execute()
    return [LegalSection(**item) for item in response.data]

def get_reporting_contacts() -> List[Dict]:
    response = supabase.table('reporting_contacts').select('*').execute()
    return response.data

def save_feedback(abuse_category: str, retrieved_section: str, rating: str, comment: str = ""):
    supabase.table('feedback').insert({
        'abuse_category': abuse_category,
        'retrieved_section': retrieved_section,
        'rating': rating,
        'comment': comment
    }).execute()