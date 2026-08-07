import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.services.classifier_service import classify_abuse_categories
from app.services.rag_service import retrieve_relevant_laws
from app.services.fact_extraction_service import extract_canonical_facts

query = "බාලවයස්කාර දරුවෙක් පස්දෙනා දෙනෙකු ලිංගික අපයෝජනයට ලක්ව ඇති බව අධිකරණ වෛද්‍ය පරීක්ෂණයකදී හෙළි වී ඇත. මෙම සිදුවීමට අදාළ නීති සහ වාර්තා කිරීමේ පියවර දැනගැනීමට අවශ්‍ය වේ."

print("Testing user second query:")
p_cat, s_cats = classify_abuse_categories(query)
facts = extract_canonical_facts(query, "si")
print(f"Primary Category: {p_cat} | Secondary: {s_cats}")
print(f"Extracted Canonical Facts: {facts}")

laws = retrieve_relevant_laws(query, abuse_category=p_cat, language="si")
print("Returned Laws:", [(l.section, l.title_en, l.relevance_score) for l in laws])
