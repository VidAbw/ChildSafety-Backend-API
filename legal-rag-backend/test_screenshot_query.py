import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.services.classifier_service import classify_abuse_categories
from app.services.rag_service import retrieve_relevant_laws
from app.services.fact_extraction_service import extract_canonical_facts

query = "දරුවෙකු වැඩිහිටි පවුලේ හිතවතෙකු විසින් වෙනත් ස්ථානයකට ගෙන ගොස් ඇඳුම් ගලවා, එම දරුවා ලිංගික හෝ වෙනත් අයුරින් ජන අතවරයකට යොදාගෙන ඇති බවට සැක පවතී. මේ සිදුවීමෙන් පසු දරුවා බියෙන්, ආරක්ෂාවක් නොමැතිව, සහ දැඩි පීඩාවෙන් සිටින බව පෙනේ. සිදුවීමට සම්බන්ධ අය දරුවා නිහඬව සිටීමට බලපෑම් කර ඇති බවත් වාර්තා වේ."

print("Analyzing screenshot query:")
p_cat, s_cats = classify_abuse_categories(query)
facts = extract_canonical_facts(query, "si")
print(f"Primary Category: {p_cat} | Secondary: {s_cats}")
print(f"Extracted Canonical Facts: {facts}")

laws = retrieve_relevant_laws(query, abuse_category=p_cat, language="si")
print("Returned Laws:", [l.section for l in laws])
