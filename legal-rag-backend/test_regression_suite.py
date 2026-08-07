import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.services.classifier_service import classify_abuse_categories
from app.services.rag_service import retrieve_relevant_laws

def run_suite():
    test_cases = [
        ("1. Physical Abuse (English)", "A 10-year-old child is repeatedly hit by a guardian, has visible injuries, and is afraid to stay at home."),
        ("2. Physical Abuse (Sinhala)", "දරුවෙකු පඩිපෙළෙන් භාරකරු කෝපයෙන් කිහිප වතාවක් පහර දී වේදනාව සහ නිල් තැල්ම ඇති කල නිවසි."),
        ("3. Sexual Abuse", "A male adult engaged in forced rape and sexual penetration of a 13-year-old minor inside a house."),
        ("4. Neglect", "Parents abandoned their 4-year-old child alone inside an apartment for three days without food or water."),
        ("5. Emotional Abuse", "A teenager is continuously shouted at, threatened, insulted, and subjected to severe emotional trauma by their caregiver."),
        ("6. Trafficking", "A gang recruited, transported, and sold a child for forced labor and debt bondage in a factory."),
        ("7. Kidnapping", "A stranger enticed and abducted a 7-year-old child away from their lawful guardianship."),
        ("8. Online / Material Abuse", "Someone uploaded and published obscene photos and videos of a minor on Telegram hosted by an internet platform."),
        ("9. Mixed Abuse (Physical + Emotional)", "A father severely beat his child with an iron rod causing bleeding injuries and threatened to kill the child if they cried."),
        ("10. Ambiguous / Non-Abuse Input", "I bought a second-hand bicycle from a shop but the seller refused to give a receipt.")
    ]

    print("\n================================================================================")
    print("RUNNING FULL LEGAL RETRIEVAL & FACT-LEVEL FILTERING REGRESSION SUITE")
    print("================================================================================\n")

    for name, query in test_cases:
        print(f"\n>>> TEST CASE: {name}")
        p_cat, s_cats = classify_abuse_categories(query)
        results = retrieve_relevant_laws(query, p_cat, language="en" if "Sinhala" not in name else "si")
        sections = [r.section for r in results]
        print(f"SUMMARY FOR '{name}': Returned Sections = {sections}\n")

if __name__ == "__main__":
    run_suite()
