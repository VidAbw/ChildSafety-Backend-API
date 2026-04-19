from pydantic import BaseModel
from typing import List

class LegalSection(BaseModel):
    id: int
    law_name: str
    section_number: str
    abuse_category: str
    legal_text_summary: str
    simple_explanation: str
    keywords: List[str]
    reporting_guidance: str
    source: str