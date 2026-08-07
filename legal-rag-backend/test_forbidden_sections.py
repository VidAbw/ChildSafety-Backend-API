import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.services.rag_service import retrieve_relevant_laws


def _collect_sections(results):
    sections = []
    for item in results:
        sections.append(item.section)
        for child in getattr(item, "related_provisions", []) or []:
            sections.append(child.section)
    return sections


def _assert_no_forbidden_sections(results, label):
    sections = _collect_sections(results)
    assert "309" not in sections, f"{label} unexpectedly returned section 309: {sections}"


def _build_frontend_fallback_results(results):
    return [
        {
            "section": item.section,
            "title": item.title,
            "related_provisions": [
                {"section": child.section, "title": child.title}
                for child in getattr(item, "related_provisions", []) or []
            ],
        }
        for item in results
    ]


def test_sinhala_general_protection_query_excludes_section_309():
    query = "දරුවා මිය ගිය පසු සිරුර සඟවා ගෙන ඇති අවස්ථාව" 
    results = retrieve_relevant_laws(query, "general_child_protection", "si")
    _assert_no_forbidden_sections(results, "Sinhala general protection query")

    fallback_results = _build_frontend_fallback_results(results)
    assert all(item["section"] != "309" for item in fallback_results)
    assert all(
        child["section"] != "309"
        for item in fallback_results
        for child in item["related_provisions"]
    )


def test_english_physical_abuse_query_excludes_section_309():
    query = "A child was beaten badly and left with bruises and injuries."
    results = retrieve_relevant_laws(query, "physical_abuse", "en")
    _assert_no_forbidden_sections(results, "English physical abuse query")


def test_english_sexual_abuse_query_excludes_section_309():
    query = "A relative took explicit photos of a child and forced the child into sexual abuse."
    results = retrieve_relevant_laws(query, "sexual_abuse", "en")
    _assert_no_forbidden_sections(results, "English sexual abuse query")


def test_sinhala_emotional_abuse_query_excludes_section_309():
    query = "දරුවාට බොහෝ විට කෑගැසී බරපතල මනෝඥාතියට පත්වූ අවස්ථාව" 
    results = retrieve_relevant_laws(query, "emotional_abuse", "si")
    _assert_no_forbidden_sections(results, "Sinhala emotional abuse query")


if __name__ == "__main__":
    test_sinhala_general_protection_query_excludes_section_309()
    test_english_physical_abuse_query_excludes_section_309()
    test_english_sexual_abuse_query_excludes_section_309()
    test_sinhala_emotional_abuse_query_excludes_section_309()
    print("All forbidden-section regression tests passed.")
