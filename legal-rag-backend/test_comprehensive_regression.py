import sys
import os
from typing import List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.services.classifier_service import classify_abuse_categories
from app.services.rag_service import retrieve_relevant_laws, LegalRetrievalResult


def run_tests():
    test_cases = [
        {
            "id": 1,
            "name": "Physical Abuse - Caregiver (English)",
            "query": "A 10-year-old child is repeatedly beaten by a guardian at home, causing visible swelling and severe pain.",
            "language": "en",
            "expect_present": ["308A"],
            "expect_absent": ["309"]
        },
        {
            "id": 2,
            "name": "Physical Abuse - Caregiver (Sinhala)",
            "query": "වයස අවුරුදු 12ක් වන දරුවෙකු නිවසේදී දරුවාගේ රැකවරණය භාරව සිටින පුද්ගලයෙකු විසින් නැවත නැවතත් පහරදීම්වලට ලක් කරනු ලබයි. මෙම පහරදීම් හේතුවෙන් දරුවාගේ ශරීරයේ තැලීම්, ඉදිමීම් සහ දැඩි ශාරීරික වේදනාවක් ඇති වී ඇත.",
            "language": "si",
            "expect_present": ["308A"],
            "expect_absent": ["309"]
        },
        {
            "id": 3,
            "name": "Physical Abuse - Stranger (English)",
            "query": "A stranger hit a 10-year-old child on the street, causing minor swelling.",
            "language": "en",
            "expect_present": ["314"],
            "expect_absent": ["308A"] # Should NOT be applicable or potential because stranger has no custody/care!
        },
        {
            "id": 4,
            "name": "Physical Abuse - Stranger (Sinhala)",
            "query": "පාරේ යන විට අමුත්තෙක් ළමයෙකුට පහර දී රිදෙව්වා.",
            "language": "si",
            "expect_present": ["314"],
            "expect_absent": ["308A"]
        },
        {
            "id": 5,
            "name": "Grievous Hurt with Weapon - Stranger (English)",
            "query": "A stranger stabbed a child with a knife causing a deep bleeding bone fracture.",
            "language": "en",
            "expect_present": ["315", "316"],
            "expect_absent": ["308A"]
        },
        {
            "id": 6,
            "name": "Near-miss Begging (English)",
            "query": "The child was crying and begging his father for food because they had nothing to eat inside the house.",
            "language": "en",
            "expect_present": ["308A"],
            "expect_absent": ["288"] # Section 288 is begging/alms, must NOT be matched!
        },
        {
            "id": 7,
            "name": "Near-miss Begging (Sinhala)",
            "query": "දරුවා කෑම ඉල්ලා හඬා වැලපෙමින් පියාගෙන් කෑම ඉල්ලා සිටියි.",
            "language": "si",
            "expect_present": ["308A"],
            "expect_absent": ["288"]
        },
        {
            "id": 8,
            "name": "Negation of Intercourse (English)",
            "query": "A relative touched a child inappropriately on the private parts, but there was no sexual intercourse or penetration.",
            "language": "en",
            "expect_present": ["365B"],
            "expect_absent": ["363", "364"] # Rape requires penetration, which is explicitly negated!
        },
        {
            "id": 9,
            "name": "Negation of Intercourse (Sinhala)",
            "query": "ඥාතියෙක් දරුවා වැරදි විදියට ස්පර්ශ කලා, නමුත් ලිංගික සංසර්ගයක් හෝ ඇතුල් කිරීමක් සිදු නොවීය.",
            "language": "si",
            "expect_present": ["365B"],
            "expect_absent": ["363", "364"]
        },
        {
            "id": 10,
            "name": "Negation of Injury (English)",
            "query": "A father slapped his child as a punishment, but there were no injuries or bodily pain at all.",
            "language": "en",
            "expect_present": [],
            "expect_absent": ["314", "315", "316"] # Slap with no injury or pain must not match hurt sections!
        },
        {
            "id": 11,
            "name": "Negation of Injury (Sinhala)",
            "query": "තාත්තා දරුවාට පහර දුන්නද කිසිදු තුවාලයක් හෝ ශාරීරික වේදනාවක් සිදු නොවීය.",
            "language": "si",
            "expect_present": [],
            "expect_absent": ["314", "315", "316"]
        },
        {
            "id": 12,
            "name": "Age boundary - 17 years old (English)",
            "query": "A 17-year-old child was severely beaten by his caregiver, causing injuries.",
            "language": "en",
            "expect_present": ["308A"],
            "expect_absent": []
        },
        {
            "id": 13,
            "name": "Age boundary - 18 years old (English)",
            "query": "An 18-year-old victim was beaten by his guardian causing injuries.",
            "language": "en",
            "expect_present": [],
            "expect_absent": ["308A"] # Must NOT match because the victim is 18 (not under 18)!
        },
        {
            "id": 14,
            "name": "Unrelated input - Civil dispute (English)",
            "query": "I bought a second-hand bicycle from a shop but the seller refused to give a receipt.",
            "language": "en",
            "expect_present": [],
            "expect_absent": ["308A", "314", "363", "288"]
        },
        {
            "id": 15,
            "name": "Unrelated input - Civil dispute (Sinhala)",
            "query": "මම කඩයකින් පාපැදියක් මිලදී ගත්තා නමුත් රිසිට්පතක් දුන්නේ නැත.",
            "language": "si",
            "expect_present": [],
            "expect_absent": ["308A", "314", "363", "288"]
        },
        {
            "id": 17,
            "name": "Unseen Scenario - Touching & Threats (English)",
            "query": "A 13-year-old child was repeatedly touched inappropriately by a known adult. The adult threatened the child to keep silent and threatened harm if the child tells anyone, causing fear and psychological distress. No explicit intercourse or penetration was stated.",
            "language": "en",
            "expect_present": ["365B", "345", "483"],
            "expect_absent": ["364A", "365", "365A", "288A", "288B"]
        },
        {
            "id": 18,
            "name": "Unseen Scenario - Touching & Threats (Sinhala)",
            "query": "වයස අවුරුදු 13ක දරුවෙකු, දන්නා වැඩිහිටියෙකු විසින් නැවත නැවතත් නුසුදුසු ලෙස ස්පර්ශ කරන ලදී. කිසිවෙකුට නොකියන ලෙස සහ පැවසුවහොත් හානියක් කරන බවට එම වැඩිහිටියා දරුවාට තර්ජනය කර ඇත. මේ හේතුවෙන් දරුවා බියට හා මානසික පීඩාවට පත්ව ඇත. කිසිදු ලිංගික සංසර්ගයක් හෝ ඇතුල් කිරීමක් ප්‍රකාශ කර නොමැත.",
            "language": "si",
            "expect_present": ["365B", "345", "483"],
            "expect_absent": ["364A", "365", "365A", "288A", "288B"]
        }
    ]

    print("\n================================================================================")
    print("RUNNING COMPREHENSIVE BILINGUAL REGRESSION TEST SUITE")
    print("================================================================================\n")

    failures = 0

    for tc in test_cases:
        name = tc["name"]
        query = tc["query"]
        lang = tc["language"]
        expect_present = tc["expect_present"]
        expect_absent = tc["expect_absent"]

        p_cat, s_cats = classify_abuse_categories(query)
        results: LegalRetrievalResult = retrieve_relevant_laws(query, p_cat, language=lang)
        
        # Collect sections returned
        returned_sections = [r.section for r in results]
        
        # Collect from applicable/potential structures
        app_secs = [x["section"] for x in results.applicable_laws]
        pot_secs = [x["section"] for x in results.potential_laws]
        active_secs = set(returned_sections + app_secs + pot_secs)

        # Verification
        passed = True
        failed_reasons = []

        for p in expect_present:
            if p not in active_secs:
                passed = False
                failed_reasons.append(f"Expected section {p} to be present, but got {active_secs}")
        
        for a in expect_absent:
            if a in active_secs:
                passed = False
                failed_reasons.append(f"Expected section {a} to be absent, but got {active_secs}")

        if passed:
            print(f"[PASS] {name} - Active: {list(active_secs)}")
        else:
            failures += 1
            print(f"[FAIL] {name} - Active: {list(active_secs)}")
            for reason in failed_reasons:
                print(f"       -> {reason}")

    print("\n================================================================================")
    if failures == 0:
        print("ALL TESTS PASSED SUCCESSFULLY!")
    else:
        print(f"{failures} TEST FAILURES DETECTED!")
    print("================================================================================\n")

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
