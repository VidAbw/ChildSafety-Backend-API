import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.services.rag_service import retrieve_relevant_laws

def test_retrieve(query, abuse_category, language):
    return retrieve_relevant_laws(query, abuse_category, language)

test_cases = [
    # 1. Simple physical abuse
    ("My teacher hit me with a stick on my back and now it is swollen and painful", "physical_abuse", "en"),
    # 2. General protection/complaint
    ("how do I make a complaint to the NCPA and who do I call?", "general_child_protection", "en"),
    # 3. Complex digital and sexual abuse
    ("My cousin took inappropriate photos of me and posted them online on Facebook, and now he is threatening to share them with my friends if I don't meet him", "sexual_abuse", "en"),
    # 4. Simple neglect
    ("My mother does not give me food and leaves me alone at home all day", "neglect", "en"),
    # 5. Sinhala physical abuse (user query)
    ("දරුවෙකු පඩිපෙළෙන් භාරකරු කෝපයෙන් කිහිප වතාවක් පහර දී වේදනාව සහ නිල් තැල්ම ඇති කල නිවසි.", "physical_abuse", "si"),
]

for idx, (query, category, lang) in enumerate(test_cases, 1):
    print(f"\n--- Test Case {idx} ({lang}) ---")
    print(f"Query: {query}")
    print(f"Detected Category: {category}")
    results = test_retrieve(query, category, lang)
    print(f"Retrieved {len(results)} parent laws:")
    for res in results:
        print(f"  - Parent: Section {res.section} ({res.law_name}): {res.title} (Score: {res.relevance_score})")
        if res.related_provisions:
            print(f"    Nested provisions ({len(res.related_provisions)}):")
            for sub in res.related_provisions:
                print(f"      * Sub-item: Section {sub.section}: {sub.title} (Score: {sub.relevance_score})")
