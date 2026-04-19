import json
import os
from typing import List
from app.schemas.legal_schema import LegalSection
from app.schemas.rag_schema import RelevantLaw

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'legal_sections.json')

def load_legal_sections() -> List[LegalSection]:
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [LegalSection(**item) for item in data]

def retrieve_relevant_laws(abuse_category: str) -> List[RelevantLaw]:
    sections = load_legal_sections()
    relevant = [s for s in sections if s.abuse_category == abuse_category]
    if not relevant:
        relevant = sections[:3]  # Default to first 3
    return [
        RelevantLaw(
            section=f"{s.law_name} {s.section_number}",
            title=s.law_name,
            simple_explanation=s.simple_explanation
        ) for s in relevant[:3]
    ]