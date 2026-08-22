# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import List, Optional

class RAGQueryRequest(BaseModel):
    description: str
    language: Optional[str] = "en"  # "en" or "si"

class RelevantLaw(BaseModel):
    section: str
    law_name: Optional[str] = None
    law_type: Optional[str] = "primary"
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
    explanation_variant: Optional[str] = None
    matched_age_rule: Optional[str] = None
    matched_legal_basis: Optional[str] = None
    related_provisions: Optional[List['RelevantLaw']] = []

try:
    RelevantLaw.model_rebuild()
except AttributeError:
    RelevantLaw.update_forward_refs()

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
    
    # Upgraded structured response fields
    incident_summary: Optional[str] = None
    facts: Optional[List[dict]] = None
    applicable_laws: Optional[List[dict]] = None
    potential_laws: Optional[List[dict]] = None
    rejected_laws: Optional[List[dict]] = None
    additional_information_needed: Optional[List[dict]] = None