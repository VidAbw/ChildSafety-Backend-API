from pydantic import BaseModel
from typing import List

class LegalSection(BaseModel):
    id: str
    law_name: str
    section_number: str
    title: str | None = None
    abuse_category: str
    legal_text_summary: str
    simple_explanation: str
    title_si: str | None = None
    simple_explanation_si: str | None = None
    reporting_guidance_si: str | None = None
    keywords: list[str]
    reporting_guidance: str
    source: str