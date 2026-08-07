import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.services.classifier_service import classify_abuse_categories
from app.services.rag_service import retrieve_relevant_laws
from app.services.fact_extraction_service import extract_canonical_facts

def run_summary_only():
    test_cases = [
        ("1. Sinhala Sexual Abuse", "ළමයෙකු පවුලේ හිතවතෙකු විසින් අනුචිත ලෙස ස්පර්ශ කළ බව පවසා ඇති අතර කිසිවෙකුට නොකියන ලෙස තර්ජනය කර ඇත.", "si"),
        ("2. English Sexual Abuse Equivalent", "A child reported inappropriate touching by a known adult relative and was threatened not to tell anyone.", "en"),
        ("3. Sinhala Physical Abuse", "දරුවෙකු පඩිපෙළෙන් භාරකරු කෝපයෙන් කිහිප වතාවක් පහර දී වේදනාව සහ නිල් තැල්ම ඇති කල නිවසි.", "si"),
        ("4. English Physical Abuse Equivalent", "A 10-year-old child is repeatedly hit by a guardian, has visible injuries, and is afraid to stay at home.", "en"),
        ("5. Sinhala Neglect", "දෙමව්පියන් තම වයස අවුරුදු 4ක දරුවා කෑම බීම නොදී දින 3ක් නිවසේ තනිව දමා ගොස් ඇත.", "si"),
        ("6. Sinhala Trafficking", "සන්නද්ධ කණ්ඩායමක් විසින් ළමයෙකු බලහත්කාරයෙන් පැහැරගෙන ගොස් වහල් සේවයේ සහ බලහත්කාර ශ්‍රමයේ යොදවා ඇත.", "si"),
        ("7. Sinhala Online Exploitation", "අන්තර්ජාලය ඔස්සේ බාලවයස්කාර දරුවෙකුගේ අසභ්‍ය ඡායාරූප සහ වීඩියෝ Telegram සමාජ මාධ්‍ය හරහා ප්‍රචාරය කර ඇත.", "si"),
        ("8. Mixed Sinhala Abuse", "පියා තම දරුවාට යකඩ පොල්ලකින් පහර දී තුවාල සිදුකර ඇති අතර කෑගැසුවොත් මරා දමන බවට තර්ජනය කර ඇත.", "si")
    ]

    print("\n" + "="*80)
    print("SUMMARY RESULTS FOR ALL 8 TEST CASES:")
    print("="*80)

    for name, query, lang in test_cases:
        p_cat, s_cats = classify_abuse_categories(query)
        canonical_facts = extract_canonical_facts(query, lang)
        results = retrieve_relevant_laws(query, p_cat, language=lang)
        sections = [r.section for r in results]
        print(f"[{name}]")
        print(f"  Query: '{query}'")
        print(f"  Language: {lang}")
        print(f"  Primary Category: {p_cat} | Secondary: {s_cats}")
        print(f"  Canonical Facts: {canonical_facts}")
        print(f"  Returned Sections: {sections}\n")

if __name__ == "__main__":
    run_summary_only()
