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


def is_supporting_law_relevant(section_id: str, query: str, abuse_category: str) -> bool:
    """
    Determines if a supporting law (e.g. NCPA Act section) is relevant to the description.
    """
    query_lower = query.lower()
    
    if section_id == "ncpa_33":
        # Section 33 is "Power to Enter and Inspect Premises"
        # Only relevant if there is physical or general abuse happening inside a premises
        premises_keywords = [
            "house", "home", "room", "premises", "inside", "locked", "building", "school", 
            "center", "place", "orphanage", "neighbor", "neighbour", "hostel", "apartment", "institution", "location",
            "gedara", "gedaraka", "kamare", "wahuwa", "waha", "hira", "koodu", "gewal", "gewala",
            "නිවස", "නිවසේ", "ගෙදර", "ගෙදරක", "කාමරය", "කාමරයක", "ගොඩනැගිල්ල", "පාසල", "ඇතුලේ", "හිරකර", "කොටු", "පරිශ්‍ර"
        ]
        is_physical_or_general = abuse_category in ["physical_abuse", "general_child_protection"]
        has_premises_keyword = any(kw in query_lower for kw in premises_keywords)
        return is_physical_or_general and has_premises_keyword

    elif section_id == "ncpa_39":
        # Section 39 is "Definition of Child Abuse"
        # Relevant for active child abuse incidents, but not for general admin/helpline questions
        general_q_keywords = [
            "how to complain", "how do i make a complaint", "make a complaint", "use the helpline", 
            "helpline", "hotline", "contact", "number", "telephone", "address", "where is", "report to",
            "complaint", "durakathana", "ankaya", "paminili",
            "පැමිණිල්ලක්", "පැමිණිලි", "ඇමතුම්", "දුරකථන", "අංකය", "කාර්යාලය", "වාර්තා"
        ]
        is_general_q = any(kw in query_lower for kw in general_q_keywords)
        return abuse_category != "general_child_protection" or not is_general_q

    return True


# Map of section_number -> group_id to cluster related sections
SECTION_GROUPS = {
    # Hurt / Grievous Hurt
    "310": "hurt_group",
    "311": "hurt_group",
    "312": "hurt_group",
    "313": "hurt_group",
    "314": "hurt_group",
    "315": "hurt_group",
    "316": "hurt_group",
    "317": "hurt_group",
    "318": "hurt_group",
    
    # Obscene material / digital CSAM
    "286A": "obscene_group",
    "286B": "obscene_group",
    
    # Rape / Grave Sexual Abuse / Sexual Offences
    "363": "rape_group",
    "364": "rape_group",
    "364A": "rape_group",
    "365": "unnatural_group",
    "365A": "gross_indecency_group",
    "365B": "grave_sexual_abuse_group",
    "365C": "privacy_group",
    
    # Kidnapping & Abduction
    "350": "kidnap_group",
    "351": "kidnap_group",
    "352": "kidnap_group",
    "353": "kidnap_group",
    "356": "kidnap_group",
    "357": "kidnap_group",
    "358": "kidnap_group",
    "358A": "kidnap_group",
}


def get_section_role(title: str, simple_explanation: str) -> str:
    """
    Classifies a section as 'punishment', 'definition', or 'offence'
    based on terminology in the title or explanation.
    """
    title_lower = title.lower()
    explanation_lower = simple_explanation.lower()
    
    if any(w in title_lower for w in ["punishment", "penalty", "sentencing"]):
        return "punishment"
    if any(w in explanation_lower for w in ["දඬුවම්", "දඬුවම", "දණ්ඩනය"]):
        return "punishment"
        
    if any(w in title_lower for w in ["definition", "defines", "meaning"]):
        return "definition"
    if "අර්ථ දැක්වීම" in explanation_lower or "අර්ථදැක්වීම" in explanation_lower or "නිර්වචනය" in explanation_lower or "රිදවීමට හේතුව" in title_lower:
        return "definition"
        
    return "offence"


def retrieve_relevant_laws(query: str, abuse_category: str, language: str, top_k: int = 3) -> List[RelevantLaw]:
    sections = load_legal_sections()
    
    # Normalize category names to the standard 6 categories
    category_normalization = {
        "trafficking_exploitation": "trafficking",
        "psychological_trauma_counseling_need": "emotional_abuse"
    }
    norm_category = category_normalization.get(abuse_category.lower(), abuse_category.lower())
    
    # Define category map for soft boosting
    category_map = {
        "sexual_abuse": ["sexual", "rape", "incest", "prostitution", "csam", "exploitation", "obscene", "assault", "harassment", "child sexual"],
        "physical_abuse": ["physical", "cruelty", "hurt", "assault", "beating", "hitting", "injury", "maltreatment", "neglect", "grievous"],
        "neglect": ["neglect", "abandonment", "exposure", "care", "without", "left alone"],
        "trafficking": ["traffic", "kidnap", "abduction", "exploitation", "slavery", "bondage", "procurer", "transport", "sold", "buying", "selling"],
        "emotional_abuse": ["emotional", "mental", "trauma", "cruelty", "shouting", "insulting", "bullying", "suffering", "harassment", "counseling", "therapy"],
        "general_child_protection": []
    }
    
    target_keywords = category_map.get(norm_category, [])
    
    # Coarse-grained category routing to map predicted category to allowed DB categories
    CATEGORY_ROUTING = {
        "sexual_abuse": ["sexual_abuse", "general_child_protection"],
        "physical_abuse": ["physical_abuse", "neglect", "general_child_protection"],
        "neglect": ["neglect", "physical_abuse", "general_child_protection"],
        "trafficking": ["trafficking", "general_child_protection"],
        "emotional_abuse": ["emotional_abuse", "general_child_protection"],
        "general_child_protection": ["physical_abuse", "sexual_abuse", "emotional_abuse", "neglect", "trafficking", "general_child_protection"]
    }

    # Strict section mapping for strict category-based filtering
    ALLOWED_SECTIONS = {
        "physical_abuse": ["308", "308A", "310", "311", "312", "313", "314", "315", "316", "317", "318", "33", "39", "286C"],
        "sexual_abuse": ["345", "363", "364", "364A", "365", "365A", "365B", "365C", "286A", "286B", "288A", "357", "358", "360A", "360B", "360E", "33", "39", "286C"],
        "neglect": ["308", "308A", "33", "39", "286C"],
        "trafficking": ["288", "288B", "350", "351", "352", "353", "354", "355", "356", "357", "358", "358A", "360A", "360C", "360D", "33", "39", "286C"],
        "emotional_abuse": ["308A", "33", "39", "286C"],
        "general_child_protection": ["286C", "309", "33", "39"]
    }

    allowed_categories = CATEGORY_ROUTING.get(norm_category, [norm_category, "general_child_protection"])

    # 1. Filter candidates: Keep primary laws that match allowed categories, and only contextually relevant supporting laws
    filtered_sections = []
    for section in sections:
        if getattr(section, "law_type", "primary") == "supporting":
            if is_supporting_law_relevant(section.id, query, norm_category):
                filtered_sections.append(section)
        else:
            # Check if category matches allowed categories for the predicted abuse category
            if not allowed_categories or section.abuse_category.lower() in [c.lower() for c in allowed_categories]:
                filtered_sections.append(section)

    if not filtered_sections:
        return []

    try:
        model = get_model()
        query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype('float32')
        
        # 2. Rank candidate sections
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
        
        # Combine and apply soft category boost and role penalties
        scored_results = []
        for i, score in enumerate(similarities):
            section = filtered_sections[i]
            
            # Check if category matches
            section_cat = section.abuse_category.lower()
            section_keywords = [k.lower() for k in section.keywords]
            
            category_match = (
                section_cat == norm_category or
                any(tk in section_cat for tk in target_keywords) or
                any(tk in k for tk in target_keywords for k in section_keywords)
            )
            
            # Substantive vs Definition/Punishment ranking adjustment (Strategy 3)
            role = get_section_role(section.title or "", section.simple_explanation)
            penalty = 0.0
            if role == "punishment":
                penalty = 0.04
            elif role == "definition":
                penalty = 0.08
                
            boosted_score = score
            if category_match:
                # Add a soft boost of +0.05 for category matching
                boosted_score += 0.05
            boosted_score -= penalty
            
            # Limit between 0.0 and 1.0
            boosted_score = max(0.0, min(float(boosted_score), 1.0))
            
            scored_results.append((boosted_score, section))
            
        # 3. Filter by threshold and category match
        RELEVANCE_THRESHOLD = 0.40
            
        strong_matches = []
        for score, section in scored_results:
            if score >= RELEVANCE_THRESHOLD:
                # D. After semantic retrieval, remove every result whose category or section number does not match the detected category
                section_cat = section.abuse_category.lower()
                section_num = section.section_number
                
                category_matches = (section_cat == norm_category)
                section_matches = (section_num in ALLOWED_SECTIONS.get(norm_category, []))
                
                if category_matches or section_matches:
                    strong_matches.append((score, section))
        
        # Sort by strongest match first
        strong_matches.sort(key=lambda x: x[0], reverse=True)
        
        # 4. Group results into parent-child structure (Strategy 1)
        grouped_results = []
        seen_groups = {}  # group_id -> parent_RelevantLaw
        
        for score, section in strong_matches:
            # Determine English title fallback
            english_title = getattr(section, "title", None) or f"{section.law_name} {section.section_number}"
            
            law_obj = RelevantLaw(
                section=section.section_number,
                law_name=section.law_name,
                law_type=getattr(section, "law_type", "primary"),
                title=section.title_si if language == "si" and getattr(section, "title_si", None) else english_title,
                title_en=english_title,
                title_si=getattr(section, "title_si", None),
                simple_explanation=section.simple_explanation_si if language == "si" and getattr(section, "simple_explanation_si", None) else section.simple_explanation,
                simple_explanation_en=section.simple_explanation,
                simple_explanation_si=getattr(section, "simple_explanation_si", None),
                reporting_guidance=section.reporting_guidance_si if language == "si" and getattr(section, "reporting_guidance_si", None) else section.reporting_guidance,
                reporting_guidance_en=section.reporting_guidance,
                reporting_guidance_si=getattr(section, "reporting_guidance_si", None),
                relevance_score=round(float(score), 3),
                related_provisions=[]
            )
            
            group_id = SECTION_GROUPS.get(section.section_number)
            if group_id:
                if group_id not in seen_groups:
                    seen_groups[group_id] = law_obj
                    grouped_results.append(law_obj)
                else:
                    parent = seen_groups[group_id]
                    if parent.related_provisions is None:
                        parent.related_provisions = []
                    parent.related_provisions.append(law_obj)
            else:
                grouped_results.append(law_obj)
                
        return grouped_results
    except Exception as e:
        print(f"Filtered search failed: {e}")
        return []