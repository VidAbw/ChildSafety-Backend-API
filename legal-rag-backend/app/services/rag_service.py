import json
import os
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.schemas.legal_schema import LegalSection
from app.schemas.rag_schema import RelevantLaw

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'legal_sections.json')
INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'legal_index.faiss')
IDS_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'ids.json')
TEXTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'texts.json')
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

_model = None
_index = None
_sections = None
_ids = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_legal_sections() -> List[LegalSection]:
    global _sections
    if _sections is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Legal sections dataset not found at {DATA_PATH}")
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _sections = [LegalSection(**item) for item in data]
    return _sections


def save_legal_sections(sections: List[LegalSection]) -> None:
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump([section.dict() for section in sections], f, ensure_ascii=False, indent=2)
    global _sections
    _sections = sections


def load_faiss_index():
    global _index, _ids
    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError('FAISS index not found. Run app/vector_store/build_index.py first.')
        _index = faiss.read_index(INDEX_PATH)
        with open(IDS_PATH, 'r', encoding='utf-8') as f:
            _ids = json.load(f)
    return _index, _ids


def build_faiss_index(sections: List[LegalSection] = None) -> None:
    if sections is None:
        sections = load_legal_sections()

    if not sections:
        raise ValueError('No legal sections available to build FAISS index.')

    model = get_model()
    texts = [
        f"{section.law_name} {section.section_number} {getattr(section, 'title', '') or ''} {section.legal_text_summary} {section.simple_explanation} {section.reporting_guidance} {section.title_si or ''} {section.simple_explanation_si or ''} {getattr(section, 'reporting_guidance_si', '') or ''}"
        for section in sections
    ]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, 0)
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(IDS_PATH, 'w', encoding='utf-8') as f:
        json.dump([section.id for section in sections], f, ensure_ascii=False, indent=2)
    with open(TEXTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    global _index, _ids
    _index = index
    _ids = [section.id for section in sections]


def import_legal_sections(sections: List[LegalSection], rebuild_index: bool = True) -> None:
    save_legal_sections(sections)
    if rebuild_index:
        build_faiss_index(sections)


def retrieve_relevant_laws(query: str, abuse_category: str, language: str, top_k: int = 3) -> List[RelevantLaw]:
    sections = load_legal_sections()
    
    # 1. Filter sections by category mapping
    filtered_sections = []
    category_match = True
    category_map = {
        "sexual abuse": ["sexual", "rape", "incest", "prostitution", "csam", "exploitation", "obscene", "assault", "harassment", "child sexual"],
        "physical abuse": ["physical", "cruelty", "hurt", "assault", "beating", "hitting", "injury", "maltreatment", "neglect", "grievous"],
        "neglect": ["neglect", "abandonment", "exposure", "care", "without", "left alone"],
        "trafficking": ["traffic", "kidnap", "abduction", "exploitation", "slavery", "bondage", "procurer", "transport", "sold", "buying", "selling"],
        "digital abuse": ["digital", "online", "computer", "photos", "videos", "internet", "social media", "platform", "csam", "material"]
    }
    
    target_keywords = category_map.get(abuse_category, [])
    
    for section in sections:
        # Check if abuse_category field or keywords match
        section_cat = section.abuse_category.lower()
        section_keywords = [k.lower() for k in section.keywords]
        
        # Broad matching: if any target keyword appears in section category or keywords
        match_found = False
        if any(tk in section_cat for tk in target_keywords) or any(tk in k for tk in target_keywords for k in section_keywords):
            match_found = True
        
        if match_found:
            filtered_sections.append(section)
        # Fallback for "general abuse" or if no match found but category is relevant
        elif abuse_category == "general abuse":
            filtered_sections.append(section)

    if not filtered_sections:
        # If no sections match the category, fall back to all sections but with a higher threshold later
        category_match = False
        filtered_sections = sections

    try:
        model = get_model()
        query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype('float32')
        
        # 2. Rank only filtered sections
        section_texts = []
        for s in filtered_sections:
            if language == "si":
                # For Sinhala, prioritize Sinhala fields to improve embedding similarity
                text = f"{getattr(s, 'title_si', '') or ''} {getattr(s, 'simple_explanation_si', '') or ''} {getattr(s, 'reporting_guidance_si', '') or ''} {s.law_name} {s.section_number} {s.legal_text_summary} {' '.join(s.keywords)}"
            else:
                text = f"{s.law_name} {s.section_number} {getattr(s, 'title', '') or ''} {s.legal_text_summary} {s.simple_explanation} {s.reporting_guidance} {' '.join(s.keywords)}"
            section_texts.append(text)
            
        section_embeddings = model.encode(section_texts, convert_to_numpy=True, show_progress_bar=False).astype('float32')
        
        # Calculate cosine similarities
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        section_norms = section_embeddings / (np.linalg.norm(section_embeddings, axis=1, keepdims=True) + 1e-9)
        similarities = np.dot(section_norms, query_norm.T).flatten()
        
        # Combine and sort
        scored_results = []
        for i, score in enumerate(similarities):
            scored_results.append((score, filtered_sections[i]))
            
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # 3. Filter by strong match threshold
        # We use a lower threshold for Sinhala to ensure valid reports are not rejected
        if language == "si":
            RELEVANCE_THRESHOLD = 0.18  # Relaxed for Sinhala to handle linguistic variations
        else:
            RELEVANCE_THRESHOLD = 0.25 if category_match else 0.40
            
        # Optional: Boost score if category matches exactly (already handled by filtering, but this ensures higher ranking)
        strong_matches = [res for res in scored_results if res[0] >= RELEVANCE_THRESHOLD]
        
        # Limit to top_k
        final_results = strong_matches[:top_k]
        
        results = []
        for score, section in final_results:
            # Determine English title fallback
            english_title = getattr(section, "title", None) or f"{section.law_name} {section.section_number}"
            
            results.append(RelevantLaw(
                section=section.section_number,
                title=section.title_si if language == "si" and getattr(section, "title_si", None) else english_title,
                title_en=english_title,
                title_si=getattr(section, "title_si", None),
                simple_explanation=section.simple_explanation_si if language == "si" and getattr(section, "simple_explanation_si", None) else section.simple_explanation,
                simple_explanation_en=section.simple_explanation,
                simple_explanation_si=getattr(section, "simple_explanation_si", None),
                reporting_guidance=section.reporting_guidance_si if language == "si" and getattr(section, "reporting_guidance_si", None) else section.reporting_guidance,
                reporting_guidance_en=section.reporting_guidance,
                reporting_guidance_si=getattr(section, "reporting_guidance_si", None),
                relevance_score=round(float(score), 3)
            ))
        return results
    except Exception as e:
        print(f"Filtered search failed: {e}")
        return []