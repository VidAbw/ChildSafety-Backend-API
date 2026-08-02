import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.services.classifier_service import classify_abuse
from app.services.rag_service import retrieve_relevant_laws

def test_system():
    print("Running classification and retrieval integration tests...")
    
    # 1. Sinhala physical-abuse description
    q1 = "දරුවන් දෙදෙනෙකුට වධ හිංසා කළ බවට චෝදනා ලැබූ පියෙකු අත්අඩංගුවට ගෙන ඇත."
    cat1 = classify_abuse(q1)
    print(f"\nQuery 1: {q1}")
    print(f"Detected Category: {cat1}")
    assert cat1 == "physical_abuse", f"Expected physical_abuse, got {cat1}"
    results1 = retrieve_relevant_laws(q1, cat1, "si")
    print(f"Retrieved {len(results1)} parent laws:")
    for res in results1:
        print(f"  - Section {res.section}: {res.title} (Score: {res.relevance_score})")
    
    # Verify allowed sections: only physical abuse, cruelty, hurt, child safety
    # Forbidden: 345, 365B, 363, 364, etc.
    forbidden = ["345", "363", "364", "364A", "365", "365A", "365B", "365C", "286A", "286B", "288A", "360B", "360E"]
    for res in results1:
        assert res.section not in forbidden, f"Forbidden section {res.section} returned for physical abuse query!"
        if res.related_provisions:
            for sub in res.related_provisions:
                assert sub.section not in forbidden, f"Forbidden nested section {sub.section} returned for physical abuse query!"

    # 2. English physical-abuse description
    q2 = "My teacher hit me with a stick on my back and now it is swollen and painful."
    cat2 = classify_abuse(q2)
    print(f"\nQuery 2: {q2}")
    print(f"Detected Category: {cat2}")
    assert cat2 == "physical_abuse", f"Expected physical_abuse, got {cat2}"
    results2 = retrieve_relevant_laws(q2, cat2, "en")
    print(f"Retrieved {len(results2)} parent laws:")
    for res in results2:
        print(f"  - Section {res.section}: {res.title} (Score: {res.relevance_score})")
    for res in results2:
        assert res.section not in forbidden, f"Forbidden section {res.section} returned for physical abuse query!"

    # 3. Sinhala sexual-abuse description
    q3 = "පියෙකු තම බාලවයස්කාර දියණිය ලිංගික අපයෝජනයට ලක් කර ඇති බවට වාර්තා වේ."
    cat3 = classify_abuse(q3)
    print(f"\nQuery 3: {q3}")
    print(f"Detected Category: {cat3}")
    assert cat3 == "sexual_abuse", f"Expected sexual_abuse, got {cat3}"
    results3 = retrieve_relevant_laws(q3, cat3, "si")
    print(f"Retrieved {len(results3)} parent laws:")
    for res in results3:
        print(f"  - Section {res.section}: {res.title} (Score: {res.relevance_score})")
    # Verify at least one sexual abuse section returned
    sexual_sections = ["345", "363", "364", "364A", "365", "365A", "365B", "365C", "286A", "286B", "288A", "360B", "360E"]
    found_sexual = False
    for res in results3:
        if res.section in sexual_sections:
            found_sexual = True
    assert found_sexual, "No sexual abuse section returned for sexual abuse query!"

    # 4. English sexual-abuse description
    q4 = "A relative took inappropriate photos of me and sexually abused me."
    cat4 = classify_abuse(q4)
    print(f"\nQuery 4: {q4}")
    print(f"Detected Category: {cat4}")
    assert cat4 == "sexual_abuse", f"Expected sexual_abuse, got {cat4}"
    results4 = retrieve_relevant_laws(q4, cat4, "en")
    print(f"Retrieved {len(results4)} parent laws:")
    for res in results4:
        print(f"  - Section {res.section}: {res.title} (Score: {res.relevance_score})")
    found_sexual = False
    for res in results4:
        if res.section in sexual_sections:
            found_sexual = True
    assert found_sexual, "No sexual abuse section returned for sexual abuse query!"

    # 5. Unrelated or unclear input
    q5 = "What is the capital of Sri Lanka?"
    cat5 = classify_abuse(q5)
    print(f"\nQuery 5: {q5}")
    print(f"Detected Category: {cat5}")
    results5 = retrieve_relevant_laws(q5, cat5, "en")
    print(f"Retrieved {len(results5)} parent laws for unrelated query.")
    assert len(results5) == 0, f"Expected 0 results for unrelated query, got {len(results5)}"

    print("\nAll integration tests passed successfully!")

if __name__ == "__main__":
    test_system()
