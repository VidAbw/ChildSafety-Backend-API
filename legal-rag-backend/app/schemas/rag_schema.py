from pydantic import BaseModel
from typing import List, Optional

class RAGQueryRequest(BaseModel):
    description: str
    language: Optional[str] = "en"  # "en" or "si"

class RelevantLaw(BaseModel):
    section: str
    title: str
    title_en: Optional[str] = None
    title_si: Optional[str] = None
    simple_explanation: str
    simple_explanation_en: Optional[str] = None
    simple_explanation_si: Optional[str] = None
    reporting_guidance: str
    reporting_guidance_en: Optional[str] = None
    reporting_guidance_si: Optional[str] = None
    relevance_score: Optional[float] = None

class RAGQueryResponse(BaseModel):
    detected_language: str
    abuse_category: str
    abuse_category_en: Optional[str] = None
    abuse_category_si: Optional[str] = None
    relevant_laws: List[RelevantLaw]
    decision_roadmap: List[str]
    decision_roadmap_en: Optional[List[str]] = None
    decision_roadmap_si: Optional[List[str]] = None
    reporting_contacts: List[dict]  # List of {"name": str, "contact": str, "description": str}
    privacy_note: str