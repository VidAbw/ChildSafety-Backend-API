import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.services.rag_service import load_legal_sections, get_model, get_section_role

query = "දරුවෙකු පඩිපෙළෙන් භාරකරු කෝපයෙන් කිහිප වතාවක් පහර දී වේදනාව සහ නිල් තැල්ම ඇති කල නිවසි."
abuse_category = "physical_abuse"
language = "si"

sections = load_legal_sections()
model = get_model()
query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype('float32')

# Define category map for soft boosting
category_map = {
    "sexual_abuse": ["sexual", "rape", "incest", "prostitution", "csam", "exploitation", "obscene", "assault", "harassment", "child sexual"],
    "physical_abuse": ["physical", "cruelty", "hurt", "assault", "beating", "hitting", "injury", "maltreatment", "neglect", "grievous"],
}
target_keywords = category_map.get(abuse_category, [])

section_texts = []
for s in sections:
    if language == "si":
        text = f"{getattr(s, 'title_si', '') or ''} {getattr(s, 'simple_explanation_si', '') or ''} {getattr(s, 'reporting_guidance_si', '') or ''} {s.law_name} {s.section_number} {s.legal_text_summary} {' '.join(s.keywords)}"
    else:
        text = f"{s.law_name} {s.section_number} {getattr(s, 'title', '') or ''} {s.legal_text_summary} {s.simple_explanation} {s.reporting_guidance} {' '.join(s.keywords)}"
    section_texts.append(text)
        
section_embeddings = model.encode(section_texts, convert_to_numpy=True, show_progress_bar=False).astype('float32')
query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
section_norms = section_embeddings / (np.linalg.norm(section_embeddings, axis=1, keepdims=True) + 1e-9)
similarities = np.dot(section_norms, query_norm.T).flatten()

print(f"Scores for all sections (sorted by boosted score):")
all_scores = []
for i, s in enumerate(sections):
    raw_score = similarities[i]
    section_cat = s.abuse_category.lower()
    section_keywords = [k.lower() for k in s.keywords]
    
    category_match = (
        section_cat == abuse_category.lower() or
        any(tk in section_cat for tk in target_keywords) or
        any(tk in k for tk in target_keywords for k in section_keywords)
    )
    
    role = get_section_role(s.title or "", s.simple_explanation)
    penalty = 0.0
    if role == "punishment":
        penalty = 0.04
    elif role == "definition":
        penalty = 0.08
        
    boosted_score = raw_score
    if category_match:
        boosted_score += 0.05
    boosted_score -= penalty
    boosted_score = max(0.0, min(float(boosted_score), 1.0))
    
    all_scores.append((boosted_score, raw_score, category_match, role, s))

all_scores.sort(key=lambda x: x[0], reverse=True)
for boosted, raw, cat_match, role, s in all_scores:
    print(f"Section {s.section_number} ({s.id}): Boosted={boosted:.3f}, Raw={raw:.3f}, CatMatch={cat_match}, Role={role}, Title={s.title}")
