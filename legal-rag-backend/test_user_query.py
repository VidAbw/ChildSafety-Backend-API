import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.services.rag_service import retrieve_relevant_laws

query = "දරුවෙකු පඩිපෙළෙන් භාරකරු කෝපයෙන් කිහිප වතාවක් පහර දී වේදනාව සහ නිල් තැල්ම ඇති කල නිවසි."
category = "physical_abuse"
results = retrieve_relevant_laws(query, category, "si")

print(f"Retrieved {len(results)} parent laws for query: {query}")
for res in results:
    print(f"  - Parent: Section {res.section} ({res.law_name}): {res.title} (Score: {res.relevance_score})")
    if res.related_provisions:
        print(f"    Nested provisions ({len(res.related_provisions)}):")
        for sub in res.related_provisions:
            print(f"      * Sub-item: Section {sub.section}: {sub.title} (Score: {sub.relevance_score})")
