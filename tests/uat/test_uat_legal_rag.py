"""
UAT Scenario 3: Legal RAG System & Guidance
Validates user expectations for the AI Legal Assistant.
Ensures parents receive contextual, cited child protection laws (Penal code) in English & Sinhala.
"""
import pytest


def test_parent_queries_child_physical_abuse_law(client):
    """
    User Story:
    As a concerned guardian, when I describe a physical abuse situation involving a caregiver,
    the Legal RAG system should retrieve relevant sections of the Penal Code (e.g. Cruelty to Children, Hurt)
    and provide actionable legal recommendations.
    """
    payload = {
        "description": "A caregiver hit a 10-year-old child repeatedly causing bruises and physical injuries. What are the legal consequences?",
        "language": "en"
    }
    response = client.post("/api/rag/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "abuse_category" in data
    assert "relevant_laws" in data
    assert "decision_roadmap" in data
    assert len(data["relevant_laws"]) > 0


def test_bilingual_sinhala_query_support(client):
    """
    User Story:
    As a Sri Lankan parent communicating in Sinhala,
    I need to ask about child neglect or cruelty in my native language and obtain accurate legal provisions.
    """
    payload = {
        "description": "දරුවෙකුට පහර දීම සහ කෲරත්වයට එරෙහිව පවතින නීති මොනවාද?",
        "language": "si"
    }
    response = client.post("/api/rag/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "relevant_laws" in data

