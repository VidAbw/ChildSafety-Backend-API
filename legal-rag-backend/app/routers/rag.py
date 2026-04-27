from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.legal_schema import LegalSection
from app.schemas.rag_schema import RAGQueryRequest, RAGQueryResponse
from app.services.language_service import detect_language
from app.services.classifier_service import classify_abuse
from app.services.rag_service import (
    retrieve_relevant_laws,
    import_legal_sections,
    build_faiss_index,
    load_legal_sections,
)
from app.services.roadmap_service import generate_roadmap
from app.services.supabase_service import get_reporting_contacts
import time

router = APIRouter()

@router.post("/api/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    start_time = time.time()

    # Detect language
    detected_lang = detect_language(request.description)

    # Classify abuse
    abuse_category = classify_abuse(request.description)

    # Retrieve relevant laws using RAG-style search
    relevant_laws = retrieve_relevant_laws(request.description, abuse_category)

    # Generate roadmap
    decision_roadmap = generate_roadmap(abuse_category)

    # Get contacts
    contacts = get_reporting_contacts()
    # Format contacts
    reporting_contacts = [
        {"name": c["name"], "contact": c["contact_number"], "description": c["description"]}
        for c in contacts
    ]

    privacy_note = "The description is processed for legal guidance and should not be stored with personal details."

    response_time = time.time() - start_time

    # For prototype, don't save feedback yet, but log
    print(f"Query processed: category={abuse_category}, response_time={response_time:.2f}s")

    return RAGQueryResponse(
        detected_language=detected_lang,
        abuse_category=abuse_category,
        relevant_laws=relevant_laws,
        decision_roadmap=decision_roadmap,
        reporting_contacts=reporting_contacts,
        privacy_note=privacy_note
    )

@router.post("/api/rag/import")
async def import_penal_code(sections: List[LegalSection], rebuild_index: bool = True):
    try:
        import_legal_sections(sections, rebuild_index=rebuild_index)
        return {"message": f"Imported {len(sections)} legal sections", "rebuild_index": rebuild_index}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/rag/rebuild-index")
async def rebuild_rag_index():
    try:
        build_faiss_index()
        return {"message": "RAG vector index rebuilt successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/rag/sections")
async def get_legal_sections():
    try:
        sections = load_legal_sections()
        return {"sections": [section.dict() for section in sections]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))