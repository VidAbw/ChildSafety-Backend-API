import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.services.classifier_service import classify_abuse
from app.services.rag_service import retrieve_relevant_laws

def test_system():
    print("\n" + "="*80)
    print("RUNNING VERIFIED 12-SCENARIO CHILD-ABUSE LEGAL RETRIEVAL INTEGRATION TEST SUITE")
    print("="*80)

    # 1. Physical Abuse
    q1 = "A father hit his 8-year-old child on the back with an iron rod causing severe bodily injury and bleeding."
    cat1 = classify_abuse(q1)
    results1 = retrieve_relevant_laws(q1, cat1, language="en")
    sec1 = [r.section for r in results1]
    print(f"\n[Test 1: Physical Abuse] Category: {cat1} | Sections: {sec1}")
    assert "308A" in sec1, f"Expected Section 308A (Cruelty/Physical harm), got {sec1}"

    # 2. Cruelty
    q2 = "A guardian willfully ill-treated, severely beat, and subjected a 10-year-old child to physical suffering and severe bodily cruelty."
    cat2 = classify_abuse(q2)
    results2 = retrieve_relevant_laws(q2, cat2, language="en")
    sec2 = [r.section for r in results2]
    print(f"\n[Test 2: Cruelty] Category: {cat2} | Sections: {sec2}")
    assert "308A" in sec2, f"Expected Section 308A for cruelty, got {sec2}"

    # 3. Neglect
    q3 = "Parents left a 6-year-old child unattended without food, water, or protection for four days."
    cat3 = classify_abuse(q3)
    results3 = retrieve_relevant_laws(q3, cat3, language="en")
    sec3 = [r.section for r in results3]
    print(f"\n[Test 3: Neglect] Category: {cat3} | Sections: {sec3}")
    assert "308" in sec3 or "308A" in sec3, f"Expected Section 308 or 308A for neglect, got {sec3}"

    # 4. Abandonment
    q4 = "A mother abandoned her 3-year-old child alone in a public bus station with intent to desert."
    cat4 = classify_abuse(q4)
    results4 = retrieve_relevant_laws(q4, cat4, language="en")
    sec4 = [r.section for r in results4]
    print(f"\n[Test 4: Abandonment] Category: {cat4} | Sections: {sec4}")
    assert "308" in sec4, f"Expected Section 308 (Abandonment), got {sec4}"

    # 5. Sexual Abuse (Incest)
    q5 = "A 14-year-old child reports that an uncle sexually touched and abused them inside their home."
    cat5 = classify_abuse(q5)
    results5 = retrieve_relevant_laws(q5, cat5, language="en")
    sec5 = [r.section for r in results5]
    print(f"\n[Test 5: Sexual Abuse / Incest] Category: {cat5} | Sections: {sec5}")
    assert any(s in ["364A", "365B", "345"] for s in sec5), f"Expected sexual abuse section, got {sec5}"

    # 6. Sexual Harassment
    q6 = "A man made unwelcome sexual comments and inappropriately touched a minor girl in public, outraging her modesty."
    cat6 = classify_abuse(q6)
    results6 = retrieve_relevant_laws(q6, cat6, language="en")
    sec6 = [r.section for r in results6]
    print(f"\n[Test 6: Sexual Harassment] Category: {cat6} | Sections: {sec6}")
    assert "345" in sec6, f"Expected Section 345 (Sexual Harassment), got {sec6}"

    # 7. Statutory Rape involving a Minor
    q7 = "An adult male engaged in forced sexual penetration and rape of a 13-year-old girl."
    cat7 = classify_abuse(q7)
    results7 = retrieve_relevant_laws(q7, cat7, language="en")
    sec7 = [r.section for r in results7]
    print(f"\n[Test 7: Statutory Rape] Category: {cat7} | Sections: {sec7}")
    assert "363" in sec7 or "364" in sec7, f"Expected Section 363/364 (Rape), got {sec7}"

    # 8. Grave Sexual Abuse
    q8 = "A person subjected a 12-year-old child to grave sexual acts and sexual violation."
    cat8 = classify_abuse(q8)
    results8 = retrieve_relevant_laws(q8, cat8, language="en")
    sec8 = [r.section for r in results8]
    print(f"\n[Test 8: Grave Sexual Abuse] Category: {cat8} | Sections: {sec8}")
    assert "365B" in sec8, f"Expected Section 365B (Grave Sexual Abuse), got {sec8}"

    # 9. Trafficking
    q9 = "A criminal gang recruited, transported, and sold a child for forced labor and debt bondage."
    cat9 = classify_abuse(q9)
    results9 = retrieve_relevant_laws(q9, cat9, language="en")
    sec9 = [r.section for r in results9]
    print(f"\n[Test 9: Trafficking] Category: {cat9} | Sections: {sec9}")
    assert "360C" in sec9 or "358A" in sec9, f"Expected Section 360C or 358A (Trafficking), got {sec9}"

    # 10. Kidnapping of Minor
    q10 = "A stranger enticed and kidnapped a 9-year-old child out of the lawful guardianship of her parents."
    cat10 = classify_abuse(q10)
    results10 = retrieve_relevant_laws(q10, cat10, language="en")
    sec10 = [r.section for r in results10]
    print(f"\n[Test 10: Kidnapping] Category: {cat10} | Sections: {sec10}")
    assert "352" in sec10, f"Expected Section 352 (Kidnapping from Lawful Guardianship), got {sec10}"

    # 11. Child Sexual Material (CSAM)
    q11 = "Someone recorded inappropriate obscene photos and videos of a minor and shared them on Telegram hosted by an internet platform."
    cat11 = classify_abuse(q11)
    results11 = retrieve_relevant_laws(q11, cat11, language="en")
    sec11 = [r.section for r in results11]
    print(f"\n[Test 11: CSAM / Online] Category: {cat11} | Sections: {sec11}")
    assert "286A" in sec11 and "286B" in sec11, f"Expected Section 286A & 286B (CSAM/ISP), got {sec11}"

    # 12. Unrelated Non-Child-Abuse Description (MUST RETURN 0 MATCHES!)
    q12 = "I bought a second-hand bicycle from a shop but the seller refused to deliver the receipt and breached the purchase contract."
    cat12 = classify_abuse(q12)
    results12 = retrieve_relevant_laws(q12, cat12, language="en")
    sec12 = [r.section for r in results12 if r.section != "INSUFFICIENT_FACTS"]
    print(f"\n[Test 12: Unrelated Non-Child-Abuse Query] Category: {cat12} | Sections: {sec12}")
    assert len(sec12) == 0, f"EXPECTED 0 MATCHES FOR UNRELATED QUERY! Got: {sec12}"

    print("\n" + "="*80)
    print("ALL 12 VERIFIED CHILD-ABUSE INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    test_system()

