import json
import os
import sys
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.schemas.legal_schema import LegalSection
from app.schemas.rag_schema import RelevantLaw
from app.services.classifier_service import classify_abuse_categories
from app.services.fact_extraction_service import extract_canonical_facts, extract_victim_age

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'legal_sections.json')
INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'legal_index.faiss')
IDS_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'ids.json')
TEXTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'texts.json')
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
FORBIDDEN_SECTIONS = {"309"}

# Whitelist mapping of verified child-abuse categories to allowed Penal Code / NCPA sections
CHILD_ABUSE_ALLOWED_SECTIONS = {
    "physical_abuse": ["308A", "315", "316", "310", "311", "312", "313", "314", "317", "318"],
    "cruelty": ["308A", "310", "311", "312", "313", "314"],
    "neglect": ["308", "288"],
    "sexual_abuse": ["345", "360E", "363", "364", "364A", "365", "365A", "365B"],
    "sexual_harassment": ["345", "365A"],
    "sexual_exploitation": ["286C", "288A", "360A", "360B"],
    "trafficking": ["288", "288B", "358A", "360C", "360D"],
    "kidnapping_abduction": ["352", "350", "351", "353", "354", "355", "356", "357", "358"],
    "online_or_material_abuse": ["286A", "286B", "365C"],
    "general_child_protection": ["308A", "308", "345", "352", "360C", "363", "365B", "39", "33", "ncpa_39", "ncpa_33"]
}

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
        filtered_data = [item for item in data if str(item.get('section_number', '')).strip() not in FORBIDDEN_SECTIONS]
        _sections = [LegalSection(**item) for item in filtered_data]
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
            raise FileNotFoundError('FAISS index not found. Run rebuild_index.py first.')
        _index = faiss.read_index(INDEX_PATH)
        with open(IDS_PATH, 'r', encoding='utf-8') as f:
            _ids = json.load(f)
    return _index, _ids


def build_faiss_index(sections: List[LegalSection] = None) -> None:
    if sections is None:
        sections = load_legal_sections()

    sections = [section for section in sections if str(getattr(section, 'section_number', '')).strip() not in FORBIDDEN_SECTIONS]

    if not sections:
        raise ValueError('No legal sections available to build FAISS index.')

    model = get_model()
    texts = [
        f"{section.law_name} {section.section_number} {getattr(section, 'title', '') or ''} {section.legal_text_summary} {section.simple_explanation} {section.reporting_guidance} {section.title_si or ''} {section.simple_explanation_si or ''} {getattr(section, 'reporting_guidance_si', '') or ''} {' '.join(section.keywords)}"
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
    sec_id_str = str(section_id).lower()
    
    if sec_id_str in ["ncpa_33", "33"]:
        premises_keywords = [
            "house", "home", "room", "premises", "inside", "locked", "building", "school", 
            "center", "place", "orphanage", "neighbor", "neighbour", "hostel", "apartment", "institution", "location",
            "gedara", "gedaraka", "kamare", "wahuwa", "waha", "hira", "koodu", "gewal", "gewala",
            "නිවස", "නිවසේ", "ගෙදර", "ගෙදරක", "කාමරය", "කාමරයක", "ගොඩනැගිල්ල", "පාසල", "ඇතුලේ", "හිරකර", "කොටු", "පරිශ්‍ර"
        ]
        is_physical_or_general = abuse_category in ["physical_abuse", "cruelty", "general_child_protection"]
        has_premises_keyword = any(kw in query_lower for kw in premises_keywords)
        return is_physical_or_general and has_premises_keyword

    elif sec_id_str in ["ncpa_39", "39"]:
        general_q_keywords = [
            "how to complain", "how do i make a complaint", "make a complaint", "use the helpline", 
            "helpline", "hotline", "contact", "number", "telephone", "address", "where is", "report to",
            "complaint", "durakathana", "ankaya", "paminili",
            "පැමිණිල්ලක්", "පැමිණිලි", "ඇමතුම්", "දුරකථන", "අංකය", "කාර්යාලය", "වාර්තා"
        ]
        is_general_q = any(kw in query_lower for kw in general_q_keywords)
        return abuse_category != "general_child_protection" or not is_general_q

    return True


SECTION_CANONICAL_REQUIREMENTS = {
    "308A": {"required_facts_any": ["physical_assault", "physical_injury", "cruelty", "neglect"]},
    "365B": {"required_facts_any": ["sexual_contact", "sexual_act"]},
    "345": {"required_facts_any": ["sexual_contact", "sexual_harassment", "sexual_act"]},
    "363": {"required_facts_any": ["penetration"]},
    "364": {"required_facts_any": ["penetration"]},
    "364A": {"required_facts_all": [["sexual_contact", "sexual_act", "penetration"], ["incest_relation"]]},
    "308": {"required_facts_any": ["abandonment", "neglect"]},
    "288": {"required_facts_any": ["begging", "neglect"]},
    "358A": {"required_facts_any": ["forced_labour", "debt_bondage", "slavery"]},
    "288B": {"required_facts_any": ["restricted_articles"]},
    "350": {"required_facts_any": ["kidnapping"]},
    "351": {"required_facts_any": ["kidnapping"]},
    "352": {"required_facts_any": ["kidnapping"]},
    "353": {"required_facts_any": ["kidnapping"]},
    "354": {"required_facts_any": ["kidnapping"]},
    "355": {"required_facts_any": ["kidnapping"]},
    "356": {"required_facts_any": ["kidnapping"]},
    "357": {"required_facts_any": ["kidnapping"]},
    "358": {"required_facts_any": ["kidnapping"]},
    "286A": {"required_facts_any": ["sexual_image_material"]},
    "286B": {"required_facts_all": [["online_contact"], ["sexual_image_material"]]},
    "286C": {"required_facts_any": ["commercial_exploitation", "sexual_contact", "sexual_act"]},
    "288A": {"required_facts_any": ["commercial_exploitation"]},
    "360A": {"required_facts_any": ["commercial_exploitation"]},
    "360B": {"required_facts_any": ["commercial_exploitation", "sexual_image_material"]},
    "360C": {"required_facts_any": ["trafficking"]},
    "360D": {"required_facts_any": ["adoption_offence"]},
    "360E": {"required_facts_any": ["sexual_contact", "sexual_act"]},
    "365": {"required_facts_any": ["sexual_act", "sexual_contact"]},
    "365A": {"required_facts_any": ["sexual_act", "sexual_contact", "sexual_harassment"]},
    "365C": {"required_facts_any": ["sexual_image_material", "online_contact"]},
    "310": {"required_facts_any": ["physical_assault", "physical_injury"]},
    "311": {"required_facts_any": ["physical_injury"]},
    "312": {"required_facts_any": ["physical_assault"]},
    "313": {"required_facts_any": ["physical_assault", "physical_injury"]},
    "314": {"required_facts_any": ["physical_assault"]},
    "315": {"required_facts_any": ["physical_assault", "physical_injury"]},
    "316": {"required_facts_any": ["physical_assault", "physical_injury"]},
    "317": {"required_facts_any": ["physical_assault"]},
    "318": {"required_facts_any": ["physical_assault", "physical_injury"]}
}

# Map of section_number -> group_id to cluster related sections
SECTION_GROUPS = {
    # Rape group
    "363": "rape_group",
    "364": "rape_group",
    # Kidnapping group
    "352": "kidnap_group",
    "350": "kidnap_group",
    "351": "kidnap_group",
    "353": "kidnap_group",
    "354": "kidnap_group",
    "355": "kidnap_group",
    "356": "kidnap_group",
    "357": "kidnap_group",
    "358": "kidnap_group",
    # Hurt group
    "308A": "hurt_group",
    "310": "hurt_group",
    "311": "hurt_group",
    "312": "hurt_group",
    "313": "hurt_group",
    "314": "hurt_group",
    "317": "hurt_group",
    "318": "hurt_group"
}


def check_fact_compatibility(
    section: LegalSection,
    query_lower: str,
    primary_category: str,
    secondary_categories: List[str],
    extracted_canonical_facts: List[str],
    victim_age: Optional[int] = None,
    fallback_mode: bool = False
) -> Tuple[bool, List[str], List[str], str]:
    """
    Validates whether user query facts satisfy legal prerequisites using Canonical Fact IDs and Victim Age.
    Supports required_facts_all, required_facts_any, optional_facts, and age limits.
    Returns (is_compatible, matched_facts, missing_facts, rejection_reason).
    """
    sec_str = str(section.section_number).strip()
    sec_cat = section.abuse_category.lower()
    all_query_categories = [primary_category] + secondary_categories

    # Age condition enforcement
    if victim_age is not None:
        if sec_str == "308" and victim_age >= 12:
            return False, [], ["victim_under_12"], f"Section 308 exposure/abandonment only applies to children under 12 years of age (victim age: {victim_age})"
        if sec_str == "308A" and victim_age >= 18:
            return False, [], ["victim_under_18"], f"Section 308A child cruelty only applies to children under 18 years of age (victim age: {victim_age})"
        if sec_str == "352" and victim_age >= 18:
            return False, [], ["minor_guardianship"], f"Section 352 kidnapping applies to minors under lawful guardianship (victim age: {victim_age})"
        if sec_str == "360A" and victim_age >= 21:
            return False, [], ["person_under_21"], f"Section 360A procuration applies to persons under 21 years of age (victim age: {victim_age})"

    # 1. Category Whitelist Check
    if primary_category != "general_child_protection":
        allowed_secs = set()
        for cat in all_query_categories:
            allowed_secs.update(CHILD_ABUSE_ALLOWED_SECTIONS.get(cat, []))

        if allowed_secs and sec_str not in allowed_secs and section.id not in allowed_secs and getattr(section, "law_type", "primary") != "supporting":
            reason = f"Section {sec_str} ({sec_cat}) not allowed for query categories: {all_query_categories}"
            return False, [], [sec_cat], reason

    if fallback_mode:
        return True, [], [], "ACCEPTED (Fallback Mode)"

    req_rules = SECTION_CANONICAL_REQUIREMENTS.get(sec_str, {})
    matched_facts = []
    missing_facts = []

    # 2. required_facts_all Check
    req_all = req_rules.get("required_facts_all") or getattr(section, 'required_facts_all', []) or []
    for req_item in req_all:
        if isinstance(req_item, list):
            found_m = [fact for fact in req_item if fact in extracted_canonical_facts]
            if found_m:
                matched_facts.extend(found_m)
            else:
                missing_facts.append(" / ".join(req_item))
        elif isinstance(req_item, str):
            if req_item in extracted_canonical_facts:
                matched_facts.append(req_item)
            else:
                missing_facts.append(req_item)

    if missing_facts:
        reason = f"Lacks mandatory required_facts_all elements ({', '.join(missing_facts)})"
        return False, list(dict.fromkeys(matched_facts)), missing_facts, reason

    # 3. required_facts_any Check
    req_any = req_rules.get("required_facts_any") or getattr(section, 'required_facts_any', []) or getattr(section, 'required_facts', []) or []
    if req_any:
        matches = [fact for fact in req_any if fact in extracted_canonical_facts]
        if matches:
            matched_facts.extend(matches)
        else:
            missing_facts = req_any
            reason = f"Missing required facts for Section {sec_str} (requires at least one of: {', '.join(req_any)})"
            return False, list(dict.fromkeys(matched_facts)), missing_facts, reason

    # 4. Sexual Offences Fact Requirement
    sexual_sections = {"345", "363", "364", "364A", "365B", "365C", "286A", "286B", "288A", "360B", "360E"}
    if sec_str in sexual_sections or sec_cat in ["sexual_abuse", "sexual_harassment", "sexual_exploitation"]:
        sexual_canonical = {"sexual_contact", "sexual_act", "penetration", "sexual_harassment", "sexual_image_material", "commercial_exploitation"}
        if not any(f in sexual_canonical for f in extracted_canonical_facts):
            reason = f"Lacks sexual abuse facts (Section {sec_str} requires sexual canonical facts)"
            return False, list(dict.fromkeys(matched_facts)), ["sexual_contact / sexual_act"], reason

    return True, list(dict.fromkeys(matched_facts)), [], "ACCEPTED"
    """
    Validates whether user query facts satisfy legal prerequisites using Canonical Fact IDs.
    Supports required_facts_all, required_facts_any, and optional_facts.
    Returns (is_compatible, matched_facts, missing_facts, rejection_reason).
    """
    sec_str = str(section.section_number).strip()
    sec_cat = section.abuse_category.lower()
    all_query_categories = [primary_category] + secondary_categories

    # 1. Category Whitelist Check
    if primary_category != "general_child_protection":
        allowed_secs = set()
        for cat in all_query_categories:
            allowed_secs.update(CHILD_ABUSE_ALLOWED_SECTIONS.get(cat, []))

        if allowed_secs and sec_str not in allowed_secs and section.id not in allowed_secs and getattr(section, "law_type", "primary") != "supporting":
            reason = f"Section {sec_str} ({sec_cat}) not allowed for query categories: {all_query_categories}"
            return False, [], [sec_cat], reason

    if fallback_mode:
        return True, [], [], "ACCEPTED (Fallback Mode)"

    req_rules = SECTION_CANONICAL_REQUIREMENTS.get(sec_str, {})
    matched_facts = []
    missing_facts = []

    # 2. required_facts_all Check
    req_all = req_rules.get("required_facts_all") or getattr(section, 'required_facts_all', []) or []
    for req_item in req_all:
        if isinstance(req_item, list):
            found_m = [fact for fact in req_item if fact in extracted_canonical_facts]
            if found_m:
                matched_facts.extend(found_m)
            else:
                missing_facts.append(" / ".join(req_item))
        elif isinstance(req_item, str):
            if req_item in extracted_canonical_facts:
                matched_facts.append(req_item)
            else:
                missing_facts.append(req_item)

    if missing_facts:
        reason = f"Lacks mandatory required_facts_all elements ({', '.join(missing_facts)})"
        return False, list(dict.fromkeys(matched_facts)), missing_facts, reason

    # 3. required_facts_any Check
    req_any = req_rules.get("required_facts_any") or getattr(section, 'required_facts_any', []) or getattr(section, 'required_facts', []) or []
    if req_any:
        matches = [fact for fact in req_any if fact in extracted_canonical_facts]
        if matches:
            matched_facts.extend(matches)
        else:
            missing_facts = req_any
            reason = f"Missing required facts for Section {sec_str} (requires at least one of: {', '.join(req_any)})"
            return False, list(dict.fromkeys(matched_facts)), missing_facts, reason

    # 4. Sexual Offences Fact Requirement
    sexual_sections = {"345", "363", "364", "364A", "365B", "365C", "286A", "286B", "288A", "360B", "360E"}
    if sec_str in sexual_sections or sec_cat in ["sexual_abuse", "sexual_harassment", "sexual_exploitation"]:
        sexual_canonical = {"sexual_contact", "sexual_act", "penetration", "sexual_harassment", "sexual_image_material", "commercial_exploitation"}
        if not any(f in sexual_canonical for f in extracted_canonical_facts):
            reason = f"Lacks sexual abuse facts (Section {sec_str} requires sexual canonical facts)"
            return False, list(dict.fromkeys(matched_facts)), ["sexual_contact / sexual_act"], reason

    return True, list(dict.fromkeys(matched_facts)), [], "ACCEPTED"


# Map of section_number -> group_id to cluster related sections
SECTION_GROUPS = {
    # Rape / Punishment for Rape
    "363": "rape_group",
    "364": "rape_group",
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
    if "අර්ථ දැක්වීම" in explanation_lower or "අර්ථදැක්වීම" in explanation_lower or "නිර්වචනය" in explanation_lower:
        return "definition"
        
    return "offence"


def retrieve_relevant_laws(query: str, abuse_category: str = None, language: str = "en", top_k: int = 3) -> List[RelevantLaw]:
    sections = load_legal_sections()
    query_lower = query.lower()
    
    # Classify primary and secondary categories from query
    primary_category, secondary_categories = classify_abuse_categories(query)
    if abuse_category and abuse_category.strip():
        req_cat = abuse_category.lower().strip()
        if req_cat != primary_category and req_cat not in secondary_categories:
            secondary_categories.insert(0, primary_category)
            primary_category = req_cat

    # Bilingual Canonical Fact Extraction & Victim Age Extraction
    extracted_canonical_facts = extract_canonical_facts(query, language)
    victim_age = extract_victim_age(query)
    fallback_mode = False

    if not extracted_canonical_facts and primary_category != "general_child_protection":
        fallback_mode = True
        print(f"[FALLBACK LOG] Canonical fact extraction produced 0 facts for category '{primary_category}'. Activating Fallback Evaluation Mode.")

    BASE_THRESHOLD = 0.35 if fallback_mode else (0.15 if language == "si" else 0.25)

    print("\n" + "="*80)
    print(f"DEBUG RETRIEVAL LOG FOR QUERY: '{query}'")
    print(f"victim_age: {victim_age}")
    print(f"query_language: {language}")
    print(f"primary_category: {primary_category}")
    print(f"secondary_categories: {secondary_categories}")
    print(f"extracted_canonical_facts: {extracted_canonical_facts}")
    print("="*80)

    candidate_sections = []
    evaluation_logs = []

    def get_age_rule_and_variant(sec_num: str, age: Optional[int]):
        if sec_num == "363":
            if age is not None and age < 16:
                return "statutory_rape_under_16", "Under 16 years of age (Statutory Rape Clause 5)"
            else:
                return "general_rape", "General / Age 16 and above (Non-consensual Rape Clauses 1-4)"
        elif sec_num == "308":
            return "abandonment_under_12", "Under 12 years of age"
        elif sec_num == "308A":
            return "child_cruelty_under_18", "Under 18 years of age"
        elif sec_num == "352":
            return "guardianship_kidnapping", "Males under 14, Females under 16"
        elif sec_num == "360A":
            return "procuration_under_21", "Under 21 years of age"
        else:
            return "standard_offence", "General"

    for section in sections:
        sec_num = str(getattr(section, 'section_number', '')).strip()
        sec_id = getattr(section, 'id', '')
        title = section.title or f"{section.law_name} {sec_num}"
        req_rules = SECTION_CANONICAL_REQUIREMENTS.get(sec_num, {})
        req_all_display = req_rules.get("required_facts_all", getattr(section, 'required_facts_all', []))
        req_any_display = req_rules.get("required_facts_any", getattr(section, 'required_facts_any', getattr(section, 'required_facts', [])))

        exp_variant, age_rule = get_age_rule_and_variant(sec_num, victim_age)

        if sec_num in FORBIDDEN_SECTIONS:
            evaluation_logs.append({
                "victim_age": victim_age,
                "query_language": language,
                "primary_category": primary_category,
                "secondary_categories": secondary_categories,
                "extracted_canonical_facts": extracted_canonical_facts,
                "section_number": sec_num,
                "matched_age_rule": age_rule,
                "explanation_variant": exp_variant,
                "required_facts_all": req_all_display,
                "required_facts_any": req_any_display,
                "matched_facts": [],
                "missing_facts": ["FORBIDDEN_SECTION"],
                "semantic_score": 0.0,
                "accepted_rejected": "rejected",
                "rejection_reason": "FORBIDDEN_SECTION"
            })
            continue

        if getattr(section, "law_type", "primary") == "supporting":
            if not is_supporting_law_relevant(sec_id, query, primary_category):
                evaluation_logs.append({
                    "victim_age": victim_age,
                    "query_language": language,
                    "primary_category": primary_category,
                    "secondary_categories": secondary_categories,
                    "extracted_canonical_facts": extracted_canonical_facts,
                    "section_number": sec_num,
                    "matched_age_rule": age_rule,
                    "explanation_variant": exp_variant,
                    "required_facts_all": req_all_display,
                    "required_facts_any": req_any_display,
                    "matched_facts": [],
                    "missing_facts": ["supporting law relevance"],
                    "semantic_score": 0.0,
                    "accepted_rejected": "rejected",
                    "rejection_reason": "Supporting law not relevant to context"
                })
                continue

        # Compatibility Check
        is_fact_ok, matched_facts, missing_facts, fact_reason = check_fact_compatibility(
            section, query_lower, primary_category, secondary_categories, extracted_canonical_facts, victim_age, fallback_mode
        )

        if not is_fact_ok:
            evaluation_logs.append({
                "victim_age": victim_age,
                "query_language": language,
                "primary_category": primary_category,
                "secondary_categories": secondary_categories,
                "extracted_canonical_facts": extracted_canonical_facts,
                "section_number": sec_num,
                "matched_age_rule": age_rule,
                "explanation_variant": exp_variant,
                "required_facts_all": req_all_display,
                "required_facts_any": req_any_display,
                "matched_facts": matched_facts,
                "missing_facts": missing_facts,
                "semantic_score": 0.0,
                "accepted_rejected": "rejected",
                "rejection_reason": fact_reason
            })
            continue

        candidate_sections.append({
            "section": section,
            "matched_facts": matched_facts,
            "req_all": req_all_display,
            "req_any": req_any_display,
            "exp_variant": exp_variant,
            "age_rule": age_rule
        })

    if not candidate_sections:
        for log in evaluation_logs:
            print(f"victim_age: {log.get('victim_age')}")
            print(f"detected_facts: {log.get('extracted_canonical_facts')}")
            print(f"section_number: {log.get('section_number')}")
            print(f"matched_age_rule: {log.get('matched_age_rule')}")
            print(f"matched_facts: {log.get('matched_facts')}")
            print(f"rejected_reason: {log.get('rejection_reason')}")
            print(f"explanation_variant: {log.get('explanation_variant')}")
            print("-" * 80)
        return []

    try:
        model = get_model()
        query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype('float32')

        # 2. Build bilingual section text representations for semantic ranking
        section_texts = []
        for item in candidate_sections:
            s = item["section"]
            title_en = getattr(s, 'title_en', '') or getattr(s, 'title', '') or ''
            title_si = getattr(s, 'title_si', '') or ''
            exp_en = getattr(s, 'simple_explanation_en', '') or getattr(s, 'simple_explanation', '') or ''
            exp_si = getattr(s, 'simple_explanation_si', '') or ''
            kws = ' '.join(s.keywords)
            summary = s.legal_text_summary or ''
            
            combined_text = f"{s.law_name} {s.section_number} {title_en} {title_si} {summary} {exp_en} {exp_si} {kws}"
            section_texts.append(combined_text)

        section_embeddings = model.encode(section_texts, convert_to_numpy=True, show_progress_bar=False).astype('float32')

        # Cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        section_norms = section_embeddings / (np.linalg.norm(section_embeddings, axis=1, keepdims=True) + 1e-9)
        similarities = np.dot(section_norms, query_norm.T).flatten()

        # 3. Compute Category Boosts, Fact Boosts & Role Penalties
        scored_candidates = []
        all_query_categories = [primary_category] + secondary_categories

        for i, raw_sim in enumerate(similarities):
            item = candidate_sections[i]
            section = item["section"]
            sec_cat = section.abuse_category.lower()

            cat_match = sec_cat in all_query_categories
            role = getattr(section, 'law_role', get_section_role(section.title or "", section.simple_explanation))
            penalty = 0.03 if role == "punishment" else (0.05 if role == "definition" else 0.0)
            boost = 0.08 if cat_match else 0.0
            fact_boost = 0.10 if item["matched_facts"] else 0.0

            final_score = max(0.0, min(float(raw_sim) + boost + fact_boost - penalty, 1.0))
            scored_candidates.append({
                "section": section,
                "matched_facts": item["matched_facts"],
                "req_all": item["req_all"],
                "req_any": item["req_any"],
                "raw_sim": float(raw_sim),
                "boost": boost,
                "penalty": penalty,
                "final_score": final_score,
                "cat_match": cat_match,
                "role": role,
                "exp_variant": item["exp_variant"],
                "age_rule": item["age_rule"]
            })

        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

        top_score = scored_candidates[0]["final_score"] if scored_candidates else 0.0
        
        # 4. Filter by Base Threshold and Dynamic Margin
        accepted_sections = []
        accepted_group_ids = set()

        for item in scored_candidates:
            section = item["section"]
            sec_num = section.section_number
            score = item["final_score"]

            min_allowed_score = max(BASE_THRESHOLD, top_score - 0.15)

            if score >= min_allowed_score:
                accepted_sections.append((score, section, item["exp_variant"], item["age_rule"]))
                group_id = SECTION_GROUPS.get(sec_num)
                if group_id:
                    accepted_group_ids.add(group_id)

        # Second pass: Accept child/secondary group sections if parent group is accepted and score >= BASE_THRESHOLD
        already_accepted_ids = {s[1].id for s in accepted_sections}
        for item in scored_candidates:
            section = item["section"]
            if section.id in already_accepted_ids:
                continue
            group_id = SECTION_GROUPS.get(section.section_number)
            if group_id and group_id in accepted_group_ids and item["final_score"] >= BASE_THRESHOLD:
                accepted_sections.append((item["final_score"], section, item["exp_variant"], item["age_rule"]))

        evaluation_logs.sort(key=lambda x: str(x["section_number"]))
        for log in evaluation_logs:
            print(f"victim_age: {log.get('victim_age')}")
            print(f"detected_facts: {log.get('extracted_canonical_facts')}")
            print(f"section_number: {log.get('section_number')}")
            print(f"matched_age_rule: {log.get('matched_age_rule')}")
            print(f"matched_facts: {log.get('matched_facts')}")
            print(f"rejected_reason: {log.get('rejection_reason')}")
            print(f"explanation_variant: {log.get('explanation_variant')}")
            print("-" * 80)

        # 5. Group parent-child structure
        grouped_results = []
        seen_groups = {}

        for score, section, variant, age_rule in accepted_sections:
            if str(getattr(section, 'section_number', '')).strip() in FORBIDDEN_SECTIONS:
                continue

            english_title = getattr(section, "title_en", None) or getattr(section, "title", None) or f"{section.law_name} {section.section_number}"

            # Dynamic explanation variant selection
            simple_exp_en = getattr(section, "simple_explanation_en", None) or getattr(section, "simple_explanation", "")
            simple_exp_si = getattr(section, "simple_explanation_si", None) or getattr(section, "simple_explanation", "")
            legal_basis = f"Penal Code Section {section.section_number}"

            if section.section_number == "363":
                if variant == "statutory_rape_under_16":
                    legal_basis = "Penal Code Section 363 Clause 5 (Statutory Rape - female under 16, consent legally irrelevant)"
                    simple_exp_en = "Under Clause 5 of Section 363, sexual intercourse with a female under 16 years of age constitutes statutory rape with or without her consent. Consent is legally irrelevant."
                    simple_exp_si = "363 වන වගන්තියේ 5 වන උපවගන්තිය යටතේ වයස අවුරුදු 16ට අඩු ගැහැණු ළමයෙකු සමඟ කැමැත්ත ඇතිව හෝ නැතිව සිදුකරන ලිංගික සංසර්ගය නීත්‍යානුකූල ස්ත්‍රී දූෂණයක් (Statutory Rape) වේ."
                else:
                    legal_basis = "Penal Code Section 363 Clauses 1-4 (Non-consensual rape committed against will or without consent)"
                    simple_exp_en = "Under Clauses 1-4 of Section 363, sexual intercourse committed against a person's will, without valid consent, or through force/coercion constitutes the offence of rape."
                    simple_exp_si = "363 වන වගන්තියේ 1-4 උපවගන්ති යටතේ පුද්ගලයෙකුගේ කැමැත්තෙන් තොරව, බලහත්කාරයෙන් හෝ තර්ජනය කර සිදුකරනු ලබන ලිංගික සංසර්ගය ස්ත්‍රී දූෂණයේ වරද වේ."

            law_obj = RelevantLaw(
                section=section.section_number,
                law_name=section.law_name,
                law_type=getattr(section, "law_type", "primary"),
                title=section.title_si if language == "si" and getattr(section, "title_si", None) else english_title,
                title_en=english_title,
                title_si=getattr(section, "title_si", None),
                simple_explanation=simple_exp_si if language == "si" and simple_exp_si else simple_exp_en,
                simple_explanation_en=simple_exp_en,
                simple_explanation_si=simple_exp_si,
                reporting_guidance=section.reporting_guidance_si if language == "si" and getattr(section, "reporting_guidance_si", None) else section.reporting_guidance,
                reporting_guidance_en=getattr(section, "reporting_guidance_en", section.reporting_guidance),
                reporting_guidance_si=getattr(section, "reporting_guidance_si", None),
                relevance_score=round(float(score), 3),
                explanation_variant=variant,
                matched_age_rule=age_rule,
                matched_legal_basis=legal_basis,
                related_provisions=[]
            )

            group_id = SECTION_GROUPS.get(section.section_number)
            is_secondary = getattr(section, "law_type", "primary") == "secondary"

            if group_id:
                if group_id not in seen_groups:
                    if not is_secondary:
                        seen_groups[group_id] = law_obj
                        grouped_results.append(law_obj)
                else:
                    parent = seen_groups[group_id]
                    if parent.related_provisions is None:
                        parent.related_provisions = []
                    parent.related_provisions.append(law_obj)
            else:
                if not is_secondary:
                    grouped_results.append(law_obj)

        print(f"FINAL RETURNED LAWS COUNT: {len(grouped_results)}")
        for g in grouped_results:
            print(f"  -> Section {g.section}: {g.title_en} (Variant: {g.explanation_variant}, Age Rule: {g.matched_age_rule}, Score: {g.relevance_score})")
            if g.related_provisions:
                for sub in g.related_provisions:
                    print(f"      * Child Section {sub.section}: {sub.title_en} (Variant: {sub.explanation_variant}, Score: {sub.relevance_score})")

        return grouped_results
    except Exception as e:
        print(f"Legal retrieval failed: {e}")
        return []