import sys
import os

sys.path.insert(0, r"c:\Users\ASUS\Documents\GitHub\ChildSafety-Backend-API\legal-rag-backend")
sys.stdout.reconfigure(encoding='utf-8')

from app.services.classifier_service import classify_abuse_categories
from app.services.fact_extraction_service import extract_canonical_facts, extract_victim_age
from app.services.rag_service import retrieve_relevant_laws

def test_age_bugfix():
    cases = [
        {
            "id": "A",
            "name": "Age 13 + Rape Facts",
            "query": "Victim age = 13. An adult male committed non-consensual sexual penetration and raped the child.",
            "exp_sec363": True,
            "exp_variant": "statutory_rape_under_16"
        },
        {
            "id": "B",
            "name": "Age 25 + Rape Facts",
            "query": "Victim age = 25. An adult male committed non-consensual sexual penetration and rape against a 25-year-old woman.",
            "exp_sec363": True,
            "exp_variant": "general_rape"
        },
        {
            "id": "C",
            "name": "Age 25 + Inappropriate Touching Only",
            "query": "Victim age = 25. Description indicates inappropriate touching and distress without intercourse.",
            "exp_sec363": False,
            "exp_variant": None
        },
        {
            "id": "D",
            "name": "Age 13 + Inappropriate Touching Only",
            "query": "Victim age = 13. Description indicates inappropriate touching and distress without intercourse.",
            "exp_sec363": False,
            "exp_variant": None
        }
    ]

    print("\n" + "="*80)
    print("RUNNING AGE-SENSITIVE SECTION 363 REGRESSION TESTS")
    print("="*80 + "\n")

    results_table = []

    for c in cases:
        print(f"--- TEST {c['id']}: {c['name']} ---")
        query = c['query']
        v_age = extract_victim_age(query)
        facts = extract_canonical_facts(query, "en")
        p_cat, s_cats = classify_abuse_categories(query)
        
        print(f"Input Query: {query}")
        print(f"Extracted Victim Age: {v_age}")
        print(f"Detected Canonical Facts: {facts}")
        print(f"Classified Category: Primary='{p_cat}', Secondary={s_cats}")
        
        laws = retrieve_relevant_laws(query, p_cat, language="en")
        sec_numbers = [l.section for l in laws]
        sec363_law = next((l for l in laws if l.section == "363"), None)
        
        passed = False
        reason = ""
        
        if c['exp_sec363']:
            if sec363_law and sec363_law.explanation_variant == c['exp_variant']:
                passed = True
                reason = f"Section 363 returned with correct explanation_variant='{sec363_law.explanation_variant}' and matched_legal_basis='{sec363_law.matched_legal_basis}'"
            elif sec363_law:
                reason = f"Section 363 returned but wrong variant: expected '{c['exp_variant']}', got '{sec363_law.explanation_variant}'"
            else:
                reason = f"Section 363 was not returned."
        else:
            if not sec363_law:
                passed = True
                reason = f"Section 363 correctly REJECTED. Returned laws: {sec_numbers}"
            else:
                reason = f"Section 363 incorrectly returned for non-rape query (variant: {sec363_law.explanation_variant})"
                
        status = "PASS" if passed else "FAIL"
        results_table.append({
            "id": c['id'],
            "name": c['name'],
            "query": query,
            "expected_sec363": "Returned (" + str(c['exp_variant']) + ")" if c['exp_sec363'] else "Rejected",
            "actual_sec363": "Returned (" + str(sec363_law.explanation_variant if sec363_law else None) + ")" if sec363_law else "Rejected",
            "status": status,
            "reason": reason
        })
        
        print(f"Test Result: {status} - {reason}\n")
        print("-" * 80 + "\n")

    print("="*80)
    print("SUMMARY OF BEFORE VS AFTER REGRESSION TESTS")
    print("="*80)
    for r in results_table:
        print(f"Test {r['id']} ({r['name']}): {r['status']}")
        print(f"   Expected: {r['expected_sec363']}")
        print(f"   Actual:   {r['actual_sec363']}")
        print(f"   Reason:   {r['reason']}\n")

if __name__ == "__main__":
    test_age_bugfix()
