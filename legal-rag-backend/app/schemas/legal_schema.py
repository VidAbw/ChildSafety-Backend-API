from pydantic import BaseModel
from typing import List, Optional, Union

class LegalSection(BaseModel):
    id: str
    law_name: str
    section_number: str
    law_type: str = "primary"
    title: Optional[str] = None
    title_en: Optional[str] = None
    title_si: Optional[str] = None
    abuse_category: str
    legal_text_summary: str
    simple_explanation: str
    simple_explanation_en: Optional[str] = None
    simple_explanation_si: Optional[str] = None
    reporting_guidance: str
    reporting_guidance_en: Optional[str] = None
    reporting_guidance_si: Optional[str] = None
    relevant_facts: List[str] = []
    required_facts: List[str] = []
    required_facts_all: List[Union[str, List[str]]] = []
    required_facts_any: List[Union[str, List[str]]] = []
    optional_facts: List[str] = []
    keywords: List[str] = []
    source: str = "Sri Lanka Penal Code"
    law_role: str = "offence"
    source_version: Optional[str] = "1.0.0"
    database_version: Optional[str] = "1.0.0"
