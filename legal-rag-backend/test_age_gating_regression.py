import sys
import os
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.services.classifier_service import classify_abuse_categories
from app.services.rag_service import retrieve_relevant_laws, LegalRetrievalResult, get_provision_age_metadata

def run_tests():
    test_cases = [
        # TEST A: Age 12 + caregiver + repeated physical assault
        {
            "id": "A_EN",
            "name": "TEST A (English) - Age 12 + caregiver + repeated physical assault",
            "query": "A 12-year-old child was repeatedly beaten by their caregiver at home, causing visible bruising and physical pain.",
            "language": "en",
            "expect_child_scope": True,
            "expect_applicable": ["308A"],
            "expect_potential": [],
            "expect_rejected": []
        },
        {
            "id": "A_SI",
            "name": "TEST A (Sinhala) - Age 12 + caregiver + repeated physical assault",
            "query": "වයස අවුරුදු 12ක් වන දරුවෙකුට එම දරුවාගේ භාරකරු විසින් නිවසේදී නැවත නැවතත් පහර දී ඇත. එම පහරදීම් හේතුවෙන් දරුවාගේ ශරීරයේ තැලීම් සහ ශාරීරික වේදනාවක් පවතී.",
            "language": "si",
            "expect_child_scope": True,
            "expect_applicable": ["308A"],
            "expect_potential": [],
            "expect_rejected": []
        },
        
        # TEST B: Age 17 + caregiver + repeated physical assault
        {
            "id": "B_EN",
            "name": "TEST B (English) - Age 17 + caregiver + repeated physical assault",
            "query": "A 17-year-old child was repeatedly beaten by their caregiver at home, causing visible bruising and physical pain.",
            "language": "en",
            "expect_child_scope": True,
            "expect_applicable": ["308A"],
            "expect_potential": [],
            "expect_rejected": []
        },
        {
            "id": "B_SI",
            "name": "TEST B (Sinhala) - Age 17 + caregiver + repeated physical assault",
            "query": "වයස අවුරුදු 17ක් වන දරුවෙකුට එම දරුවාගේ භාරකරු විසින් නිවසේදී නැවත නැවතත් පහර දී ඇත. එම පහරදීම් හේතුවෙන් දරුවාගේ ශරීරයේ තැලීම් සහ ශාරීරික වේදනාවක් පවතී.",
            "language": "si",
            "expect_child_scope": True,
            "expect_applicable": ["308A"],
            "expect_potential": [],
            "expect_rejected": []
        },

        # TEST C: Age 18 + physical assault
        {
            "id": "C_EN",
            "name": "TEST C (English) - Age 18 + physical assault",
            "query": "An 18-year-old was beaten, causing visible bruising and physical pain.",
            "language": "en",
            "expect_child_scope": False,
            "expect_applicable": ["314"],
            "expect_potential": [],
            "expect_rejected": ["308A"]
        },
        {
            "id": "C_SI",
            "name": "TEST C (Sinhala) - Age 18 + physical assault",
            "query": "වයස අවුරුදු 18ක් වන පුද්ගලයෙකුට පහර දී ඇති අතර එමඟින් ශරීරයේ තැලීම් සහ ශාරීරික වේදනාවක් සිදුවී ඇත.",
            "language": "si",
            "expect_child_scope": False,
            "expect_applicable": ["314"],
            "expect_potential": [],
            "expect_rejected": ["308A"]
        },

        # TEST D: Age 24 + repeated physical assault
        {
            "id": "D_EN",
            "name": "TEST D (English) - Age 24 + repeated physical assault",
            "query": "A 24-year-old was repeatedly beaten by their caregiver at home, causing visible bruising and physical pain.",
            "language": "en",
            "expect_child_scope": False,
            "expect_applicable": ["314"],  # General Penal Code sections only
            "expect_potential": [],
            "expect_rejected": ["308A", "286C", "33", "39"] # All child-specific sections rejected
        },
        {
            "id": "D_SI",
            "name": "TEST D (Sinhala) - Age 24 + repeated physical assault (CURRENT FAILING INPUT)",
            "query": "වයස අවුරුදු 24 ක් වන දරුවෙකුගේ රැකවරණය භාරව සිටින වැඩිහිටි පුද්ගලයෙකු දරුවාට නිවසේදී නැවත නැවතත් අතින් පහර දී ඇත. එම පහරදීම් හේතුවෙන් දරුවාගේ ශරීරයේ තැලීම් ඇති වී ඇති අතර ශාරීරික වේදනාවක්ද පවතී.",
            "language": "si",
            "expect_child_scope": False,
            "expect_applicable": ["314"],
            "expect_potential": [],
            "expect_rejected": ["308A", "286C", "33", "39"]
        },

        # TEST E: No age stated + caregiver + physical assault
        {
            "id": "E_EN",
            "name": "TEST E (English) - No age stated + caregiver + physical assault",
            "query": "A child was beaten by their caregiver at home, causing visible bruising and physical pain.",
            "language": "en",
            "expect_child_scope": None, # UNKNOWN
            "expect_applicable": ["314"], # General provisions should still be fully applicable
            "expect_potential": ["308A"], # Child-specific provisions must be potential and request age
            "expect_rejected": []
        },
        {
            "id": "E_SI",
            "name": "TEST E (Sinhala) - No age stated + caregiver + physical assault",
            "query": "දරුවෙකුගේ රැකවරණය භාරව සිටින වැඩිහිටි පුද්ගලයෙකු දරුවාට නිවසේදී අතින් පහර දී ඇත. එම පහරදීම් හේතුවෙන් දරුවාගේ ශරීරයේ තැලීම් ඇති වී ඇති අතර ශාරීරික වේදනාවක්ද පවතී.",
            "language": "si",
            "expect_child_scope": None, # UNKNOWN
            "expect_applicable": ["314"],
            "expect_potential": ["308A"],
            "expect_rejected": []
        }
    ]

    print("\n================================================================================")
    print("RUNNING AGE GATE & LEGAL SCOPE ROUTING REGRESSION TEST SUITE")
    print("================================================================================\n")

    failures = 0

    for tc in test_cases:
        name = tc["name"]
        query = tc["query"]
        lang = tc["language"]
        expect_child_scope = tc["expect_child_scope"]
        expect_applicable = tc["expect_applicable"]
        expect_potential = tc["expect_potential"]
        expect_rejected = tc["expect_rejected"]

        print(f"\n--------------------------------------------------------------------------------")
        print(f"Executing: {name}")
        print(f"Query: {query}")
        print(f"--------------------------------------------------------------------------------")

        p_cat, s_cats = classify_abuse_categories(query)
        results: LegalRetrievalResult = retrieve_relevant_laws(query, p_cat, language=lang)

        # Collect results for validation
        app_secs = [x["section"] for x in results.applicable_laws]
        pot_secs = [x["section"] for x in results.potential_laws]
        rej_secs = [x["section"] for x in results.rejected_laws]

        # Verify child scope
        facts_list = results.facts
        victim_age = None
        for f in facts_list:
            if f["fact"] == "victim_age":
                victim_age = f["value"]
                break
        
        derived_victim_under_18 = "UNKNOWN"
        if victim_age is not None:
            derived_victim_under_18 = True if victim_age < 18 else False

        passed = True
        errors = []

        # Validate child scope expected value
        if expect_child_scope is True:
            if derived_victim_under_18 is not True:
                passed = False
                errors.append(f"Expected child scope = True, but derived {derived_victim_under_18}")
        elif expect_child_scope is False:
            if derived_victim_under_18 is not False:
                passed = False
                errors.append(f"Expected child scope = False, but derived {derived_victim_under_18}")
        elif expect_child_scope is None:
            if derived_victim_under_18 != "UNKNOWN":
                passed = False
                errors.append(f"Expected child scope = UNKNOWN, but derived {derived_victim_under_18}")

        # Validate expected applicable sections
        for sec in expect_applicable:
            if sec not in app_secs:
                passed = False
                errors.append(f"Expected Section {sec} to be APPLICABLE, but got applicable: {app_secs}, potential: {pot_secs}")

        # Validate expected potential sections
        for sec in expect_potential:
            if sec not in pot_secs:
                passed = False
                errors.append(f"Expected Section {sec} to be POTENTIAL, but got applicable: {app_secs}, potential: {pot_secs}")

        # Validate expected rejected sections (must either be in rejected list or completely absent from active)
        for sec in expect_rejected:
            if sec in app_secs or sec in pot_secs:
                passed = False
                errors.append(f"Expected Section {sec} to be REJECTED/ABSENT, but it is active in results!")

            # Check if reason is properly recorded for child specific rejections
            if sec == "308A" and expect_child_scope is False:
                found_rej = False
                for r in results.rejected_laws:
                    if r["section"] == "308A":
                        found_rej = True
                        expected_reason = "Victim does not satisfy the statutory under-18 requirement."
                        if r["reason"] != expected_reason:
                            passed = False
                            errors.append(f"Expected rejection reason for 308A to be '{expected_reason}', but got '{r['reason']}'")
                if not found_rej:
                    passed = False
                    errors.append("Section 308A was not found in the rejected_laws list!")

        if passed:
            print(f"[PASS] {tc['id']} passed successfully.")
        else:
            failures += 1
            print(f"[FAIL] {tc['id']} failed. Errors:")
            for err in errors:
                print(f"  - {err}")

    print("\n================================================================================")
    if failures == 0:
        print("ALL AGE GATING REGRESSION TESTS PASSED SUCCESSFULLY!")
    else:
        print(f"{failures} TEST FAILURES DETECTED!")
    print("================================================================================\n")

    if failures > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
