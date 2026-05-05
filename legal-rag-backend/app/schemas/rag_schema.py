from pydantic import BaseModel
from typing import List, Optional

class RAGQueryRequest(BaseModel):
    description: str
    language: Optional[str] = "en"  # "en" or "si"

class RelevantLaw(BaseModel):
    section: str
    title: str
    simple_explanation: str
    reporting_guidance: str
    relevance_score: Optional[float] = None

class RAGQueryResponse(BaseModel):
    detected_language: str
    abuse_category: str
    relevant_laws: List[RelevantLaw]
    decision_roadmap: List[str]
    reporting_contacts: List[dict]  # List of {"name": str, "contact": str, "description": str}
    privacy_note: str