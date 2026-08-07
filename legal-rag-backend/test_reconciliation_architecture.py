import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.services.classifier_service import classify_abuse_categories
from app.services.rag_service import retrieve_relevant_laws, FORBIDDEN_SECTIONS

PRIMARY_ALLOWED_SECTIONS = {
    "286A", "286B", "286C", "288", "288A", "288B", "308", "308A",
    "315", "316", "345", "352", "358A", "360A", "360B", "360C",
    "360D", "360E", "363", "364A", "365", "365A", "365B", "365C",
    "39", "33", "ncpa_39", "ncpa_33"
}

SECONDARY_SECTIONS = {
    "310", "311", "312", "313", "314", "317", "318",
    "350", "351", "353", "354", "355", "356", "357", "358", "364"
}

def test_primary_sections_appear_as_top_level():
    print(">>> TEST 1: Primary sections appear as top-level results...")
    query = "A guardian severely beat a child causing injury and fractures."
    p_cat, _ = classify_abuse_categories(query)
    results = retrieve_relevant_laws(query, p_cat, "en")
    top_sections = [r.section for r in results]
    print(f"    Returned top-level sections: {top_sections}")
    assert any(s in ["308A", "315", "316"] for s in top_sections), "Primary physical abuse section missing!"
    print("    PASSED!\n")

def test_secondary_sections_never_appear_as_standalone_top_level():
    print(">>> TEST 2: Secondary sections never appear as standalone top-level results...")
    queries = [
        ("Physical abuse query", "Child hit by guardian causing pain", "physical_abuse"),
        ("Sexual abuse query", "Inappropriate touching of a minor", "sexual_abuse"),
        ("Kidnapping query", "Stranger took minor away from lawful guardianship", "kidnapping_abduction"),
        ("Trafficking query", "Child sold for forced labor", "trafficking")
    ]
    for label, query, cat in queries:
        results = retrieve_relevant_laws(query, cat, "en")
        top_sections = [r.section for r in results]
        for sec in top_sections:
            assert sec not in SECONDARY_SECTIONS, f"{label} returned secondary section '{sec}' as standalone top-level result!"
    print("    PASSED!\n")

def test_punishment_sections_grouped_under_parent():
    print(">>> TEST 3: Punishment sections grouped under parent offence...")
    query = "A minor girl was subjected to non-consensual sexual penetration and rape."
    results = retrieve_relevant_laws(query, "sexual_abuse", "en")
    top_sections = [r.section for r in results]
    assert "363" in top_sections, "Section 363 (Rape) should be in top-level results!"
    sec_363_obj = next(r for r in results if r.section == "363")
    child_secs = [c.section for c in (sec_363_obj.related_provisions or [])]
    print(f"    Parent Section 363 child provisions: {child_secs}")
    assert "364" in child_secs, "Section 364 (Punishment for rape) should be grouped under Section 363!"
    assert "364" not in top_sections, "Section 364 must NOT be a standalone top-level result!"
    print("    PASSED!\n")

def test_bilingual_equivalence():
    print(">>> TEST 4: Sinhala and English versions of same case return equivalent legal results...")
    en_query = "A relative inappropriately touched a child and threatened them."
    si_query = "ළමයෙකු පවුලේ ඥාතියෙකු විසින් අනුචිත ලෙස ස්පර්ශ කළ අතර තර්ජනය කර ඇත."
    
    en_results = retrieve_relevant_laws(en_query, "sexual_abuse", "en")
    si_results = retrieve_relevant_laws(si_query, "sexual_abuse", "si")
    
    en_secs = set(r.section for r in en_results)
    si_secs = set(r.section for r in si_results)
    
    print(f"    English returned top sections: {en_secs}")
    print(f"    Sinhala returned top sections: {si_secs}")
    assert "365B" in en_secs and "365B" in si_secs, "Both EN and SI must return Section 365B (Grave sexual abuse)!"
    print("    PASSED!\n")

def test_section_309_never_returned():
    print(">>> TEST 5: Section 309 is never returned...")
    test_queries = [
        "A child attempted to harm themselves after emotional abuse.",
        " Suicide or depression inquiry.",
        "දරුවා බියට පත්වී සිටින අවස්ථාව"
    ]
    for q in test_queries:
        results = retrieve_relevant_laws(q, "general_child_protection", "en")
        all_secs = []
        for r in results:
            all_secs.append(r.section)
            all_secs.extend([c.section for c in (r.related_provisions or [])])
        assert "309" not in all_secs, f"Section 309 unexpectedly returned for query '{q}'"
    print("    PASSED!\n")

def run_all_tests():
    print("\n================================================================================")
    print("RUNNING RECONCILED ARCHITECTURE VERIFICATION TEST SUITE")
    print("================================================================================\n")
    test_primary_sections_appear_as_top_level()
    test_secondary_sections_never_appear_as_standalone_top_level()
    test_punishment_sections_grouped_under_parent()
    test_bilingual_equivalence()
    test_section_309_never_returned()
    print("ALL RECONCILIATION ARCHITECTURE TESTS PASSED SUCCESSFULLY!\n")

if __name__ == "__main__":
    run_all_tests()
