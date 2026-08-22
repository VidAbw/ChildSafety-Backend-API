import json
import os
import sys
from typing import List, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.schemas.legal_schema import LegalSection
from app.schemas.rag_schema import RelevantLaw
from app.services.classifier_service import classify_abuse_categories
from app.services.fact_extraction_service import extract_canonical_facts, extract_victim_age, extract_all_structured_facts as extract_all_structured_facts_imported

class LegalRetrievalResult(list):
    def __init__(self, items, incident_summary="", facts=None, applicable_laws=None, potential_laws=None, rejected_laws=None, additional_information_needed=None):
        super().__init__(items)
        self.incident_summary = incident_summary
        self.facts = facts or []
        self.applicable_laws = applicable_laws or []
        self.potential_laws = potential_laws or []
        self.rejected_laws = rejected_laws or []
        self.additional_information_needed = additional_information_needed or []


DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'legal_sections.json')
INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'legal_index.faiss')
IDS_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'ids.json')
TEXTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'texts.json')
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
FORBIDDEN_SECTIONS = {"309"}

# Set DEBUG_RETRIEVAL_LOG=true to enable verbose per-section diagnostics
_DEBUG_LOG = os.getenv("DEBUG_RETRIEVAL_LOG", "false").strip().lower() == "true"

CHILD_ABUSE_ALLOWED_SECTIONS = {
    "physical_abuse": ["308A", "315", "316", "310", "311", "312", "313", "314", "317", "318", "483", "486"],
    "cruelty": ["308A", "310", "311", "312", "313", "314"],
    "neglect": ["308", "288", "308A"],

    "sexual_abuse": ["345", "360E", "363", "364", "364A", "365", "365A", "365B", "483", "486"],
    "sexual_harassment": ["345", "365A"],
    "sexual_exploitation": ["286C", "288A", "360A", "360B"],
    "trafficking": ["288", "288B", "358A", "360C", "360D"],
    "kidnapping_abduction": ["352", "350", "351", "353", "354", "355", "356", "357", "358"],
    "online_or_material_abuse": ["286A", "286B", "365C"],
    "general_child_protection": ["308A", "308", "345", "352", "360C", "363", "365B", "39", "33", "ncpa_39", "ncpa_33", "483", "486"]
}

LEGAL_KNOWLEDGE_BASE = {
    "308A": {
        "section": "308A",
        "title": "Cruelty to children",
        "jurisdiction": "Sri Lanka",
        "law": "Penal Code",
        "required_elements": [
            "victim_under_18",
            "offender_has_custody_charge_or_care",
            "wilful_assault_or_ill_treatment_or_neglect_or_abandonment",
            "conduct_likely_to_cause_suffering_or_injury_to_health"
        ],
        "supporting_facts": [
            "physical_injury",
            "mental_trauma",
            "repeated_abuse",
            "pain",
            "bruising",
            "swelling"
        ],
        "negative_conditions": [
            "victim_18_or_older"
        ],
        "source": "Sri Lankan Penal Code",
        "source_version": "Amendment Act No. 22 of 1995"
    },
    "314": {
        "section": "314",
        "title": "Punishment for voluntarily causing hurt",
        "jurisdiction": "Sri Lanka",
        "law": "Penal Code",
        "required_elements": [
            "voluntarily_causing_hurt",
            "causing_bodily_pain_or_disease_or_infirmity"
        ],
        "supporting_facts": [
            "physical_assault",
            "pain",
            "bruising",
            "swelling",
            "injury"
        ],
        "negative_conditions": [
            "no_physical_contact",
            "no_injury_occurred"
        ],
        "source": "Sri Lankan Penal Code",
        "source_version": "Original Penal Code Ordinance No. 2 of 1883"
    },
    "315": {
        "section": "315",
        "title": "Voluntarily causing hurt by dangerous weapons or means",
        "jurisdiction": "Sri Lanka",
        "law": "Penal Code",
        "required_elements": [
            "voluntarily_causing_hurt",
            "causing_bodily_pain_or_disease_or_infirmity",
            "use_of_dangerous_weapon_or_dangerous_means"
        ],
        "supporting_facts": [
            "physical_assault",
            "shooting_instrument",
            "stabbing_instrument",
            "cutting_instrument",
            "weapon_likely_to_cause_death",
            "fire_or_heated_substance",
            "poison_or_corrosive",
            "explosive_substance",
            "harmful_substance",
            "dangerous_animal"
        ],
        "negative_conditions": [
            "no_weapon_was_used",
            "no_physical_contact",
            "no_injury_occurred"
        ],
        "source": "Sri Lankan Penal Code",
        "source_version": "Original Penal Code Ordinance No. 2 of 1883"
    },
    "316": {
        "section": "316",
        "title": "Voluntarily causing grievous hurt by dangerous weapons or means",
        "jurisdiction": "Sri Lanka",
        "law": "Penal Code",
        "required_elements": [
            "voluntarily_causing_hurt",
            "satisfying_statutory_grievous_hurt_category"
        ],
        "supporting_facts": [
            "permanent_loss_of_sight",
            "permanent_loss_of_hearing",
            "loss_of_member_or_joint",
            "permanent_impairment_of_member_or_joint",
            "permanent_disfiguration_of_head_or_face",
            "fracture_or_dislocation_of_bone_or_tooth",
            "injury_endangering_life",
            "severe_bodily_pain_duration_20_days",
            "unable_to_follow_ordinary_pursuits_duration_20_days",
            "qualifying_surgery"
        ],
        "negative_conditions": [
            "no_physical_contact",
            "no_injury_occurred"
        ],
        "source": "Sri Lankan Penal Code",
        "source_version": "Original Penal Code Ordinance No. 2 of 1883"
    },
    "308": {
        "section": "308",
        "title": "Exposure and abandonment of a child under twelve years",
        "jurisdiction": "Sri Lanka",
        "law": "Penal Code",
        "required_elements": [
            "victim_under_12",
            "parent_or_person_having_care",
            "intent_to_wholly_abandon"
        ],
        "supporting_facts": [
            "abandoned_in_public",
            "deserted",
            "left_alone"
        ],
        "negative_conditions": [
            "victim_12_or_older"
        ],
        "source": "Sri Lankan Penal Code",
        "source_version": "Original Penal Code Ordinance No. 2 of 1883"
    },
    "288": {
        "section": "288",
        "title": "Causing or procuring children to beg",
        "jurisdiction": "Sri Lanka",
        "law": "Penal Code",
        "required_elements": [
            "victim_child",
            "causing_or_procuring_to_beg"
        ],
        "supporting_facts": [
            "begging",
            "alms",
            "beggar"
        ],
        "negative_conditions": [],
        "source": "Sri Lankan Penal Code",
        "source_version": "Original Penal Code Ordinance No. 2 of 1883"
    },
    "345": {
        "section": "345",
        "title": "Sexual harassment",
        "jurisdiction": "Sri Lanka",
        "law": "Penal Code",
        "required_elements": [
            "sexual_harassment_conduct",
            "outraging_modesty_or_sexual_advances"
        ],
        "supporting_facts": [
            "sexual_comments",
            "unwanted_sexual_touching",
            "unwelcome_conduct",
            "catcall"
        ],
        "negative_conditions": [
            "no_sexual_conduct"
        ],
        "source": "Sri Lankan Penal Code",
        "source_version": "Amendment Act No. 22 of 1995"
    },
    "363": {
        "section": "363",
        "title": "Rape (Statutory Rape)",
        "jurisdiction": "Sri Lanka",
        "law": "Penal Code",
        "required_elements": [
            "sexual_intercourse",
            "against_will_or_without_consent_or_under_16"
        ],
        "supporting_facts": [
            "penetration",
            "statutory_rape",
            "forced_sex"
        ],
        "negative_conditions": [
            "without_intercourse",
            "no_penetration"
        ],
        "source": "Sri Lankan Penal Code",
        "source_version": "Amendment Act No. 22 of 1995"
    },
    "364": {
        "section": "364",
        "title": "Punishment for rape",
        "jurisdiction": "Sri Lanka",
        "law": "Penal Code",
        "required_elements": [
            "rape_conviction"
        ],
        "supporting_facts": [
            "penetration"
        ],
        "negative_conditions": [
            "without_intercourse",
            "no_penetration"
        ],
        "source": "Sri Lankan Penal Code",
        "source_version": "Amendment Act No. 22 of 1995"
    },
    "364A": {
        "section": "364A",
        "title": "Incest",
        "jurisdiction": "Sri Lanka",
        "law": "Penal Code",
        "required_elements": [
            "sexual_intercourse_or_grave_abuse",
            "incestuous_relationship_parent_relative"
        ],
        "supporting_facts": [
            "uncle",
            "father",
            "brother",
            "stepfather",
            "relative"
        ],
        "negative_conditions": [
            "without_intercourse",
            "no_penetration"
        ],
        "source": "Sri Lankan Penal Code",
        "source_version": "Amendment Act No. 22 of 1995"
    },
    "365B": {
        "section": "365B",
        "title": "Grave sexual abuse",
        "jurisdiction": "Sri Lanka",
        "law": "Penal Code",
        "required_elements": [
            "grave_sexual_conduct_short_of_intercourse"
        ],
        "supporting_facts": [
            "sexual_contact",
            "sexual_act",
            "touching_private_parts",
            "groping"
        ],
        "negative_conditions": [
            "no_sexual_conduct"
        ],
        "source": "Sri Lankan Penal Code",
        "source_version": "Amendment Act No. 22 of 1995"
    }
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
    "365B": {"required_facts_any": ["sexual_contact", "sexual_act", "sexual_touching"]},
    "345": {"required_facts_any": ["sexual_contact", "sexual_harassment", "sexual_act", "sexual_touching"]},
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
    "318": {"required_facts_any": ["physical_assault", "physical_injury"]},
    "483": {"required_facts_any": ["threats", "threat_of_harm", "threat_to_keep_silent"]},
    "486": {"required_facts_any": ["threats", "threat_of_harm", "threat_to_keep_silent"]}
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
    "318": "hurt_group",
    # Intimidation group
    "483": "intimidation_group",
    "486": "intimidation_group"
}


def has_dangerous_weapon_or_means(query_lower: str) -> bool:
    # 1. Check English keywords
    english_weapons = [
        "weapon", "weapons", "gun", "pistol", "firearm", "knife", "knives", "blade", "blades", 
        "sword", "swords", "dagger", "axe", "machete", "iron rod", "metal bar", "hammer", "bat", 
        "fire", "poison", "acid", "corrosive", "chemical", "noxious", "explosive", "boiling water", 
        "hot water", "pour hot", "stick", "sticks", "pole", "poles"
    ]
    if any(kw in query_lower for kw in english_weapons):
        return True

    # 2. Check Sinhala keywords
    sinhala_weapons = [
        "ආයුධ", "පිහි", "කඩු", "තුවක්කු", "යකඩ පොල්ල", "ගිනි", "ගින්දර", "ගින්න", "ඇසිඩ්", "රසායනික", 
        "පුපුරණ", "පොල්ල", "කෝටු"
    ]
    if any(kw in query_lower for kw in sinhala_weapons):
        return True

    # 3. Check Sinhala poison safely (excluding house "නිවස", day "දවස", year "වසර", etc.)
    if "විෂ" in query_lower:
        # Check it is not just "subject" (විෂය)
        clean_q = query_lower.replace("විෂය", "")
        if "විෂ" in clean_q:
            return True
    if "වස" in query_lower:
        clean_q = query_lower
        for false_positive in ["නිවස", "නිවසේ", "දවස", "දවසේ", "වයස", "අවසර", "අවසාන", "වසර", "ජීවත්වන", "අවස්ථා"]:
            clean_q = clean_q.replace(false_positive, "")
        if "වස" in clean_q:
            return True

    return False



def has_grievous_hurt_elements(query_lower: str) -> bool:
    grievous_keywords = [
        "emasculation", "impotent", "castration", "නපුංසක", "වන්ධ්‍යා",
        "blind", "sight", "deaf", "hearing", "අන්ධ", "පෙනීම", "බිහිරි", "ඇසීම",
        "limb", "joint", "amputation", "amputate", "severed", "අතපය", "අත් පා", "සන්ධි",
        "disfigure", "disfigurement", "scar", "facial", "විකෘති",
        "fracture", "fractured", "dislocate", "dislocated", "bone broken", "broken bone", 
        "broken tooth", "teeth broken", "tooth knocked", "knocked out tooth", 
        "බිඳී", "බිඳීම", "බිඳීම්", "පැනීම", "කැඩී", "හැලී",
        "endanger life", "endangers life", "life-threatening", "critical condition", "icu", "coma", 
        "20 days", "twenty days", "දින 20", "දවස් 20", "තර්ජන", "මරණාසන්න"
    ]
    return any(kw in query_lower for kw in grievous_keywords)


def has_custody_charge_care(query_lower: str) -> bool:
    relationship_keywords = [
        "caregiver", "care", "custody", "charge", "guardian", "parent", "parents", "father", 
        "mother", "dad", "mom", "uncle", "aunt", "relative", "relatives", "family", "stepfather", 
        "stepmother", "teacher", "warden", "nanny", "babysitter", "maid", "housekeeper", 
        "adult in charge", "taking care of", "responsible for", "in charge of",
        "භාරව", "රැකවරණය", "භාරකාර", "භාරකරු", "මව", "පියා", "තාත්තා", "අම්මා", "දෙමාපිය", 
        "දෙමව්පිය", "ඥාති", "මාමා", "නැන්දා", "ගුරු", "රැකබලා"
    ]
    return any(kw in query_lower for kw in relationship_keywords)


def has_assault_or_ill_treatment(query_lower: str) -> bool:
    assault_keywords = [
        "hit", "beat", "beaten", "struck", "assault", "assaulted", "slap", "slapped", "punch", 
        "punched", "kick", "kicked", "beating", "physically harmed", "harm", "abuse", "abused", 
        "ill-treat", "ill-treated", "ill-treatment", "cruelty", "cruel", "punish", "punished", 
        "punishment", "corporal punishment", "torture",
        "පහර", "ගැහුවා", "ගහනවා", "ගහලා", "බැට", "හිංසනය", "හිංසා", "කෲර", "දඬුවම්", "නොසලකා"
    ]
    return any(kw in query_lower for kw in assault_keywords)

def has_suffering_or_injury(query_lower: str) -> bool:
    suffering_keywords = [
        "suffering", "injury", "injured", "wound", "wounded", "bleeding", "fracture", "bruise", 
        "bruising", "swelling", "pain", "visible injuries", "harm", "damage", "hurt", "distress", "agony",
        "තුවාල", "ලේ ගැලීම", "ලේ", "තැල්ම", "තැලීම්", "ඉදිමීම්", "ඉදිමුම්", "වේදනාව", "වේදනා", 
        "පීඩා", "දුක්", "හානි", "රිදෙනවා"
    ]
    return any(kw in query_lower for kw in suffering_keywords)


def extract_negative_facts(query_lower: str) -> dict:
    negatives = {}
    
    # Check for sexual penetration negation
    penetration_neg_patterns = [
        r"\b(?:no|without|did not|never)\s+(?:explicit\s+|sexual\s+)?(?:intercourse|penetration|penetrating|rape)\b",
        r"\b(?:intercourse|penetration|penetrating|rape)\s+(?:was not|did not|never occurred|is absent|was stated)\b",
        r"no\s+(?:explicit\s+)?(?:intercourse|penetration)\s+or\s+(?:explicit\s+)?(?:intercourse|penetration)",
        r"no\s+explicit\s+(?:intercourse|penetration|penetrating|rape)",
        r"(?:intercourse|penetration)\s+was\s+not\s+stated",
        r"without\s+any\s+(?:intercourse|penetration)",
        r"ලිංගික සංසර්ගයක් සිදු නොවීය",
        r"සංසර්ගයක් සිදු නොවීය",
        r"ඇතුල් කිරීමක් සිදු නොවීය",
        r"ලිංගික සංසර්ගයක් හෝ ඇතුල් කිරීමක් ප්‍රකාශ කර නොමැත",
        r"ලිංගික සංසර්ගයක් හෝ ඇතුල් කිරීමක් සිදු නොවීය"
    ]
    is_negated = False
    for pat in penetration_neg_patterns:
        if re.search(pat, query_lower):
            is_negated = True
            break
    if is_negated:
        negatives["penetration"] = False
        negatives["sexual_intercourse"] = False
        negatives["intercourse"] = False

    # Check for weapon negation
    weapon_negations = [
        "no weapon", "no weapon was used", "did not use a weapon", "did not use a knife", 
        "without weapon", "ආයුධ භාවිතා කළේ නැත", "ආයුධයක් තිබුණේ නැත", "without any weapon"
    ]
    if any(neg in query_lower for neg in weapon_negations):
        negatives["use_of_dangerous_weapon_or_dangerous_means"] = False

    # Check for injury negation
    injury_negations = [
        "no injury", "no injury occurred", "no physical contact", "there was no physical contact",
        "තුවාල සිදු නොවීය", "ශාරීරික ස්පර්ශයක් සිදු නොවීය", "without any injury", "without injuries"
    ]
    if any(neg in query_lower for neg in injury_negations):
        negatives["physical_injury"] = False
        negatives["physical_contact"] = False

    return negatives


def extract_all_structured_facts(query: str, language: str) -> dict:
    facts = extract_all_structured_facts_imported(query, language)
    # Add is_minor for backward compatibility
    victim_age = facts.get("victim_age")
    is_minor = None
    if victim_age is not None:
        is_minor = victim_age < 18
    else:
        query_lower = query.lower()
        child_kws = [
            "child", "minor", "boy", "girl", "kid", "toddler", "infant", "under 18", "year-old", 
            "student", "son", "daughter", "schoolchild", "දරුවා", "දරුවෙකු", "ළමයා", "ළමයෙකු", 
            "කුඩා දරුවා", "බාලවයස්කාර", "පුතා", "දුව"
        ]
        if any(kw in query_lower for kw in child_kws):
            is_minor = True
    facts["is_minor"] = is_minor
    return facts


def evaluate_legal_elements(sec_num: str, facts: dict) -> Tuple[dict, str]:
    elements = {}
    
    # Context elements (age, relationship) evaluate to SATISFIED, NOT_SATISFIED, or UNKNOWN
    def eval_context(val: Optional[bool]) -> str:
        if val is True: return "SATISFIED"
        if val is False: return "NOT_SATISFIED"
        return "UNKNOWN"

    # Core conduct/injury elements MUST be True to be SATISFIED, otherwise they are NOT_SATISFIED
    def eval_conduct(val: Optional[bool]) -> str:
        if val is True: return "SATISFIED"
        return "NOT_SATISFIED"

    if sec_num == "308A":
        elements["victim_under_18"] = eval_context(facts.get("is_minor"))
        elements["offender_has_custody_charge_or_care"] = eval_context(facts.get("custody_or_care"))
        elements["wilful_assault_ill_treatment_neglect_abandonment"] = eval_conduct(
            facts.get("physical_assault") is True or facts.get("neglect") is True or facts.get("abandonment") is True
        )
        elements["conduct_likely_to_cause_suffering_or_injury"] = eval_conduct(
            facts.get("health_suffering") is True or facts.get("physical_injury") is True or facts.get("physical_assault") is True or facts.get("neglect") is True or facts.get("abandonment") is True or facts.get("food_deprivation") is True
        )

    elif sec_num == "308":
        is_u12 = None
        if facts.get("victim_age") is not None:
            is_u12 = facts["victim_age"] < 12
        else:
            is_u12 = facts.get("is_minor")
        elements["victim_under_12"] = eval_context(is_u12)
        elements["parent_or_person_having_care"] = eval_context(facts.get("custody_or_care"))
        elements["abandonment_or_exposure"] = eval_conduct(facts.get("abandonment"))
        elements["intent_to_wholly_abandon"] = eval_conduct(facts.get("intent_to_wholly_abandon"))

    elif sec_num in ["310", "312", "314"]:
        elements["voluntarily_causing_hurt"] = eval_conduct(facts.get("physical_assault"))
        elements["causing_bodily_pain_disease_infirmity"] = eval_conduct(facts.get("physical_injury"))

    elif sec_num == "311":
        # Definition section: satisfied for educational reference if physical injury is present
        elements["grievous_hurt_definition_reference"] = eval_context(facts.get("physical_injury"))

    elif sec_num == "313":
        elements["voluntarily_causing_hurt"] = eval_conduct(facts.get("physical_assault"))
        elements["grievous_hurt_category_satisfied"] = eval_conduct(facts.get("injury_severity") == "grievous")

    elif sec_num == "315":
        elements["voluntarily_causing_hurt"] = eval_conduct(facts.get("physical_assault"))
        elements["causing_bodily_pain_disease_infirmity"] = eval_conduct(facts.get("physical_injury"))
        elements["use_of_dangerous_weapon_or_means"] = eval_conduct(facts.get("weapon_or_dangerous_means"))

    elif sec_num == "316":
        elements["voluntarily_causing_hurt"] = eval_conduct(facts.get("physical_assault"))
        elements["grievous_hurt_category_satisfied"] = eval_conduct(facts.get("injury_severity") == "grievous")
        elements["use_of_dangerous_weapon_or_means"] = eval_conduct(facts.get("weapon_or_dangerous_means"))

    elif sec_num == "317":
        elements["voluntarily_causing_hurt"] = eval_conduct(facts.get("physical_assault"))
        elements["causing_bodily_pain_disease_infirmity"] = eval_conduct(facts.get("physical_injury"))
        elements["extortion_coercion_intent"] = eval_conduct(facts.get("threats"))

    elif sec_num == "318":
        elements["voluntarily_causing_hurt"] = eval_conduct(facts.get("physical_assault"))
        elements["grievous_hurt_category_satisfied"] = eval_conduct(facts.get("injury_severity") == "grievous")
        elements["extortion_coercion_intent"] = eval_conduct(facts.get("threats"))

    elif sec_num in ["363", "364"]:
        # Rape elements: penetration + against will or without consent or under 16
        elements["sexual_intercourse_penetration"] = eval_conduct(facts.get("penetration"))
        
        is_u16 = None
        if facts.get("victim_age") is not None:
            is_u16 = facts["victim_age"] < 16
        else:
            is_u16 = facts.get("is_minor")
            
        elements["against_will_or_without_consent_or_under_16"] = eval_context(
            facts.get("against_will") is True or 
            facts.get("consent") is False or 
            is_u16 is True or 
            facts.get("threats") is True
        )

    elif sec_num == "364A":
        elements["incestuous_sexual_intercourse"] = eval_conduct(
            facts.get("penetration") is True or facts.get("intercourse") is True
        )
        elements["incestuous_relationship"] = eval_context(facts.get("offender_relationship") in ["parent", "relative"])

    elif sec_num == "365B":
        # Grave sexual abuse: sexual touching or act, does not amount to rape
        elements["grave_sexual_conduct"] = eval_conduct(
            facts.get("sexual_touching") is True or facts.get("sexual_contact") is True or facts.get("sexual_act") is True
        )
        
        # Verify if it amounts to rape under 363
        is_u16_rape = False
        if facts.get("victim_age") is not None and facts.get("victim_age") < 16:
            is_u16_rape = True
        elif facts.get("victim_age") is None and facts.get("is_minor") is True:
            is_u16_rape = True
            
        amounts_to_rape = (facts.get("penetration") is True or facts.get("intercourse") is True) and (
            facts.get("against_will") is True or 
            facts.get("consent") is False or 
            is_u16_rape or 
            facts.get("threats") is True
        )
        elements["does_not_amount_to_rape"] = eval_conduct(not amounts_to_rape)

    elif sec_num == "345":
        elements["sexual_harassment_conduct"] = eval_conduct(
            facts.get("sexual_harassment") is True or facts.get("sexual_touching") is True or facts.get("sexual_contact") is True or facts.get("threats") is True
        )

    elif sec_num == "350":
        # Definition of kidnapping
        elements["kidnapping_definition_reference"] = eval_context(facts.get("kidnapping"))

    elif sec_num == "351":
        # Kidnapping from Sri Lanka: kidnapping conduct + taking beyond limits
        elements["kidnapping_or_abduction_conduct"] = eval_conduct(facts.get("kidnapping") is True or facts.get("abduction") is True)
        elements["taking_out_of_sri_lanka"] = eval_context(facts.get("online_contact") is not True) # Fallback to context or not online

    elif sec_num == "352":
        # Kidnapping from lawful guardianship: Males < 14, Females < 16
        is_minor_kidnap = None
        if facts.get("victim_sex") == "female":
            is_minor_kidnap = facts["victim_age"] < 16 if facts.get("victim_age") is not None else facts.get("is_minor")
        elif facts.get("victim_sex") == "male":
            is_minor_kidnap = facts["victim_age"] < 14 if facts.get("victim_age") is not None else facts.get("is_minor")
        else:
            is_minor_kidnap = facts["victim_age"] < 16 if facts.get("victim_age") is not None else facts.get("is_minor")
            
        elements["victim_minor_under_guardianship"] = eval_context(is_minor_kidnap)
        elements["taking_or_enticing_from_guardian"] = eval_conduct(facts.get("taking_from_guardian") is True or facts.get("kidnapping") is True)

    elif sec_num == "353":
        # Abduction definition: force or deceit
        elements["compelled_by_force_or_deceit"] = eval_conduct(facts.get("abduction"))

    elif sec_num == "354":
        # Punishment for kidnapping
        elements["kidnapping_punishment_reference"] = eval_conduct(facts.get("kidnapping") is True or facts.get("abduction") is True)

    elif sec_num == "355":
        # Kidnapping or abducting to murder
        elements["kidnapping_or_abduction_conduct"] = eval_conduct(facts.get("kidnapping") is True or facts.get("abduction") is True)
        elements["murder_intent"] = eval_context(facts.get("threats")) # Proxy

    elif sec_num == "356":
        # Kidnapping or abducting to confine
        elements["kidnapping_or_abduction_conduct"] = eval_conduct(facts.get("kidnapping") is True or facts.get("abduction") is True)
        elements["wrongful_confinement_intent"] = eval_conduct(facts.get("confinement"))

    elif sec_num == "357":
        # Kidnapping or abducting female to compel marriage
        elements["kidnapping_or_abduction_conduct"] = eval_conduct(facts.get("kidnapping") is True or facts.get("abduction") is True)
        elements["female_victim"] = eval_context(facts.get("victim_sex") == "female")

    elif sec_num == "358":
        # Kidnapping to subject to grievous hurt or slavery
        elements["kidnapping_or_abduction_conduct"] = eval_conduct(facts.get("kidnapping") is True or facts.get("abduction") is True)
        elements["grievous_hurt_or_slavery_intent"] = eval_conduct(facts.get("injury_severity") == "grievous" or facts.get("trafficking") is True)

    elif sec_num == "358A":
        elements["forced_labour_or_slavery_conduct"] = eval_conduct(
            facts.get("trafficking") is True or facts.get("commercial_exploitation") is True or facts.get("begging") is True
        )

    elif sec_num == "360A":
        is_u21 = None
        if facts.get("victim_age") is not None:
            is_u21 = facts["victim_age"] < 21
        else:
            is_u21 = facts.get("is_minor")
        elements["victim_under_21"] = eval_context(is_u21)
        elements["procuring_conduct"] = eval_conduct(facts.get("commercial_exploitation"))

    elif sec_num == "360B":
        elements["exploitation_conduct"] = eval_conduct(facts.get("commercial_exploitation") is True or facts.get("sexual_image_material") is True)
        elements["victim_child"] = eval_context(facts.get("is_minor"))

    elif sec_num == "360C":
        elements["trafficking_conduct"] = eval_conduct(facts.get("trafficking"))

    elif sec_num == "360D":
        elements["adoption_offence_conduct"] = eval_conduct(facts.get("trafficking"))

    elif sec_num == "360E":
        elements["soliciting_conduct"] = eval_conduct(facts.get("commercial_exploitation") is True or facts.get("sexual_contact") is True or facts.get("sexual_act") is True)
        elements["victim_child"] = eval_context(facts.get("is_minor"))

    elif sec_num == "365":
        elements["unnatural_carnal_intercourse"] = eval_conduct(
            facts.get("unnatural_intercourse") is True or facts.get("sodomy") is True
        )

    elif sec_num == "365A":
        elements["gross_indecency_conduct"] = eval_conduct(facts.get("gross_indecency") is True)

    elif sec_num in ["286A", "286B", "365C"]:
        elements["obscene_material_conduct"] = eval_conduct(facts.get("sexual_image_material"))
        if sec_num == "286B":
            elements["computer_or_online_means"] = eval_context(facts.get("online_contact"))

    elif sec_num == "286C":
        elements["premises_used_for_abuse"] = eval_conduct(facts.get("confinement") is True or facts.get("sexual_act") is True or facts.get("physical_assault") is True)

    elif sec_num == "288":
        elements["begging_conduct"] = eval_conduct(facts.get("begging"))
        elements["victim_child"] = eval_context(facts.get("is_minor"))

    elif sec_num == "288A":
        elements["employ_child_as_procurer"] = eval_conduct(facts.get("employ_child_as_procurer") is True)
        elements["victim_child"] = eval_context(facts.get("is_minor"))

    elif sec_num == "288B":
        elements["traffic_restricted_articles"] = eval_conduct(facts.get("traffic_restricted_articles") is True)
        elements["victim_child"] = eval_context(facts.get("is_minor"))

    elif sec_num == "483":
        elements["threat_to_cause_alarm_or_force_omission"] = eval_conduct(
            facts.get("threat_to_keep_silent") is True or facts.get("threat_of_harm") is True or facts.get("threats") is True
        )
        elements["criminal_intimidation_threat"] = eval_conduct(
            facts.get("threat_of_harm") is True or facts.get("threats") is True
        )

    elif sec_num == "486":
        elements["criminal_intimidation_punishment_trigger"] = eval_conduct(
            facts.get("threat_to_keep_silent") is True or facts.get("threat_of_harm") is True or facts.get("threats") is True
        )
        elements["criminal_intimidation_threat"] = eval_conduct(
            facts.get("threat_of_harm") is True or facts.get("threats") is True
        )

    elif sec_num == "39":
        # NCPA Act child abuse broad definition
        elements["child_abuse_definition_reference"] = eval_context(facts.get("is_minor"))

    elif sec_num == "33":
        # NCPA power to enter/inspect
        elements["premises_inspection_reference"] = eval_context(facts.get("confinement") is True or facts.get("custody_or_care") is True)

    else:
        elements["general_offence_elements_satisfied"] = "SATISFIED"

    status = "strong_match"
    if any(val == "NOT_SATISFIED" for val in elements.values()):
        status = "rejected"
    elif any(val == "UNKNOWN" for val in elements.values()):
        status = "potential_match"
        
    return elements, status




def check_fact_compatibility(
    section: LegalSection,
    query_lower: str,
    primary_category: str,
    secondary_categories: List[str],
    extracted_canonical_facts: List[str],
    victim_age: Optional[int] = None,
    fallback_mode: bool = False,
    status: str = None
) -> Tuple[bool, List[str], List[str], str]:
    """
    Validates whether user query facts satisfy legal prerequisites using Canonical Fact IDs and Victim Age.
    """
    sec_str = str(section.section_number).strip()
    sec_cat = section.abuse_category.lower()

    # Enforce age limits
    if victim_age is not None:
        if sec_str == "308" and victim_age >= 12:
            return False, [], ["victim_under_12"], f"Section 308 exposure/abandonment only applies to children under 12 years of age (victim age: {victim_age})"
        if sec_str == "308A" and victim_age >= 18:
            return False, [], ["victim_under_18"], f"Section 308A child cruelty only applies to children under 18 years of age (victim age: {victim_age})"
        if sec_str == "352" and victim_age >= 18:
            return False, [], ["minor_guardianship"], f"Section 352 kidnapping applies to minors under lawful guardianship (victim age: {victim_age})"
        if sec_str == "360A" and victim_age >= 21:
            return False, [], ["person_under_21"], f"Section 360A procuration applies to persons under 21 years of age (victim age: {victim_age})"

    return True, [], [], "ACCEPTED"


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


def compile_evidence_and_reason(sec_num: str, facts: dict, elements: dict) -> Tuple[List[str], str]:
    evidence = []
    reasons = []
    
    if facts.get("is_minor") is True:
        evidence.append("Victim is under 18 years of age.")
        reasons.append("victim is a minor")
    if facts.get("custody_or_care") is True:
        evidence.append("Offender has custody, charge, or care of the victim.")
        reasons.append("offender is caregiver/guardian")
    if facts.get("physical_assault") is True:
        evidence.append("Wilful physical assault/beating is described.")
        reasons.append("physical assault occurred")
    if facts.get("sexual_contact") is True:
        evidence.append("Sexual touching/abuse is described.")
        reasons.append("sexual touching occurred")
    if facts.get("penetration") is True:
        evidence.append("Sexual penetration/intercourse is described.")
        reasons.append("sexual penetration occurred")
    if facts.get("neglect") is True:
        evidence.append("Neglect/lack of essential care is described.")
        reasons.append("neglect occurred")
    if facts.get("abandonment") is True:
        evidence.append("Abandonment/desertion is described.")
        reasons.append("abandonment occurred")
    if facts.get("weapon_or_dangerous_means") is True:
        evidence.append("Dangerous weapon or dangerous means was used.")
        reasons.append("dangerous weapon/means was used")
    if facts.get("physical_injury") is True:
        evidence.append("Physical injury, pain, bruising, or swelling is described.")
        reasons.append("physical injury/pain caused")
    if facts.get("injury_severity") == "grievous":
        evidence.append("Grievous injury (fracture, dislocation, disfigurement, or life-endangering hurt) is described.")
        reasons.append("statutory grievous hurt caused")
    
    reason_str = "The statutory elements are supported because: " + ", ".join(reasons) + "."
    return evidence, reason_str


def generate_incident_summary(facts: dict, language: str) -> str:
    if language == "si":
        summary_parts = []
        victim_part = "දරුවෙකු" if facts.get("is_minor") else "වැඩිහිටියෙකු"
        if facts.get("victim_sex") == "female":
            victim_part += " (ගැහැණු)"
        elif facts.get("victim_sex") == "male":
            victim_part += " (පිරිමි)"
            
        summary_parts.append(f"මෙම සිද්ධියට {victim_part} සම්බන්ධ වේ.")
        
        if facts.get("offender_relationship"):
            rel_map = {
                "parent": "මව/පියා",
                "guardian": "භාරකරු",
                "caregiver": "රැකබලා ගන්නා තැනැත්තා",
                "teacher": "ගුරුවරයා",
                "stranger": "නුහුරු පුද්ගලයෙකු",
                "employer": "ස්වාමියා",
                "relative": "ඥාතියෙකු"
            }
            rel_name = rel_map.get(facts["offender_relationship"], "පුද්ගලයෙකු")
            summary_parts.append(f"චූදිත පුද්ගලයා වනුයේ {rel_name} වේ.")
        else:
            summary_parts.append("චූදිතයා සහ විපතට පත්වූ තැනැත්තා අතර සම්බන්ධය පැහැදිලි නැත.")
            
        conducts = []
        if facts.get("physical_assault"):
            conducts.append("ශාරීරික පහරදීම්")
        if facts.get("sexual_contact"):
            conducts.append("අනිසි ලිංගික ස්පර්ශයන්")
        if facts.get("penetration") is True:
            conducts.append("ලිංගික දූෂණය/සංසර්ගය")
        elif facts.get("penetration") is False:
            conducts.append("ලිංගික සංසර්ගයෙන් තොර ලිංගික ක්‍රියා")
        if facts.get("neglect"):
            conducts.append("නොසලකා හැරීම")
        if facts.get("abandonment"):
            conducts.append("අත්හැර දැමීම")
            
        if conducts:
            summary_parts.append(f"සිද්ධි විස්තරයට අනුව {', '.join(conducts)} සිදු කර ඇත.")
            
        if facts.get("weapon_or_dangerous_means") is True:
            summary_parts.append("සිද්ධිය සඳහා අනතුරුදායක ආයුධයක් හෝ උපක්‍රමයක් භාවිතා කර ඇත.")
        elif facts.get("weapon_or_dangerous_means") is False:
            summary_parts.append("කිසිදු ආයුධයක් භාවිතා කර නොමැති බව පැහැදිලිව සඳන් වේ.")
            
        injuries = []
        if facts.get("physical_injury") is True:
            injuries.append("ශාරීරික වේදනාව සහ තැලීම්")
        if facts.get("injury_severity") == "grievous":
            injuries.append("බරපතල තුවාල/අස්ථි බිඳීම්")
            
        if injuries:
            summary_parts.append(f"එහි ප්‍රතිඵලයක් ලෙස විපතට පත්වූ තැනැත්තා {', '.join(injuries)} වලට ලක්ව ඇත.")
            
        return " ".join(summary_parts)
    else:
        summary_parts = []
        victim_part = "a child" if facts.get("is_minor") else "an adult"
        if facts.get("victim_sex") == "female":
            victim_part += " (female)"
        elif facts.get("victim_sex") == "male":
            victim_part += " (male)"
            
        summary_parts.append(f"The incident involves {victim_part}.")
        
        if facts.get("offender_relationship"):
            summary_parts.append(f"The alleged offender is a {facts['offender_relationship']}.")
            if facts.get("custody_or_care"):
                summary_parts.append("The offender has custody or care of the victim.")
        else:
            summary_parts.append("The relationship of the offender to the victim is not fully specified.")
            
        conducts = []
        if facts.get("physical_assault"):
            conducts.append("physical assault/beating")
        if facts.get("sexual_contact"):
            conducts.append("unwanted sexual touching")
        if facts.get("penetration") is True:
            conducts.append("sexual penetration/intercourse")
        elif facts.get("penetration") is False:
            conducts.append("sexual contact without penetration")
        if facts.get("neglect"):
            conducts.append("wilful neglect")
        if facts.get("abandonment"):
            conducts.append("abandonment")
            
        if conducts:
            summary_parts.append(f"The description alleges {', '.join(conducts)}.")
            
        if facts.get("weapon_or_dangerous_means") is True:
            summary_parts.append("A dangerous weapon or dangerous means was used during the incident.")
        elif facts.get("weapon_or_dangerous_means") is False:
            summary_parts.append("It is explicitly stated that no weapon was used.")
            
        injuries = []
        if facts.get("physical_injury") is True:
            injuries.append("physical pain and bruising")
        if facts.get("injury_severity") == "grievous":
            injuries.append("statutory grievous hurt")
            
        if injuries:
            summary_parts.append(f"As a result, the victim suffered {', '.join(injuries)}.")
            
        return " ".join(summary_parts)


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

    # 1. Bilingual Structured Fact Extraction
    facts = extract_all_structured_facts(query, language)
    extracted_canonical_facts = extract_canonical_facts(query, language)
    victim_age = facts["victim_age"]
    fallback_mode = False

    if not extracted_canonical_facts and primary_category != "general_child_protection":
        fallback_mode = True

    if not extracted_canonical_facts and primary_category == "general_child_protection":
        return LegalRetrievalResult([], incident_summary=generate_incident_summary(facts, language), facts=[], applicable_laws=[], potential_laws=[], rejected_laws=[], additional_information_needed=[])

    BASE_THRESHOLD = 0.35 if fallback_mode else (0.15 if language == "si" else 0.25)


    candidate_sections = []
    
    # Track rejected sections for final response
    rejected_laws_list = []
    additional_info_list = []

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
        req_rules = SECTION_CANONICAL_REQUIREMENTS.get(sec_num, {})
        req_all_display = req_rules.get("required_facts_all", getattr(section, 'required_facts_all', []))
        req_any_display = req_rules.get("required_facts_any", getattr(section, 'required_facts_any', getattr(section, 'required_facts', [])))

        exp_variant, age_rule = get_age_rule_and_variant(sec_num, victim_age)

        if sec_num in FORBIDDEN_SECTIONS:
            continue

        if getattr(section, "law_type", "primary") == "supporting":
            if not is_supporting_law_relevant(sec_id, query, primary_category):
                continue

        # Evaluate legal elements satisfaction and contradiction
        elements, status = evaluate_legal_elements(sec_num, facts)
        
        # Double check legacy check_fact_compatibility (passing status)
        is_fact_ok, matched_facts, missing_facts, fact_reason = check_fact_compatibility(
            section, query_lower, primary_category, secondary_categories, extracted_canonical_facts, victim_age, fallback_mode, status=status
        )
        
        # If legacy compatibility check fails, let's mark it rejected or potential based on whether elements are contradicted
        if not is_fact_ok:
            if status != "rejected":
                status = "rejected"
                
        # If status is rejected, log to rejected list
        if status == "rejected":
            rejected_reason = fact_reason if not is_fact_ok else "Mandatory legal elements or factual prerequisites were not satisfied."
            rejected_laws_list.append({
                "section": sec_num,
                "reason": rejected_reason
            })
            continue

        candidate_sections.append({
            "section": section,
            "matched_facts": matched_facts,
            "req_all": req_all_display,
            "req_any": req_any_display,
            "exp_variant": exp_variant,
            "age_rule": age_rule,
            "elements": elements,
            "status": status
        })

    # If no candidate sections, we return empty LegalRetrievalResult
    if not candidate_sections:
        return LegalRetrievalResult([], incident_summary=generate_incident_summary(facts, language), facts=[], applicable_laws=[], potential_laws=[], rejected_laws=rejected_laws_list, additional_information_needed=[])

    try:
        model = get_model()
        query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype('float32')

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

        scored_candidates = []
        all_query_categories = [primary_category] + secondary_categories

        for i, raw_sim in enumerate(similarities):
            item = candidate_sections[i]
            section = item["section"]
            sec_num = str(section.section_number).strip()
            sec_cat = section.abuse_category.lower()
            status = item["status"]
            elements = item["elements"]

            # Calculate score using exact element status tiers
            raw_sim = float(raw_sim)
            if status == "strong_match":
                final_score = 0.90 + 0.10 * raw_sim
                match_level = "EXACT"
            elif status == "potential_match":
                satisfied_count = sum(1 for val in elements.values() if val == "SATISFIED")
                total_elements = len(elements)
                element_ratio = satisfied_count / total_elements if total_elements > 0 else 1.0
                if element_ratio >= 0.5:
                    final_score = 0.75 + 0.14 * raw_sim
                    match_level = "STRONG"
                else:
                    final_score = 0.50 + 0.24 * raw_sim
                    match_level = "PARTIAL"
            else:
                final_score = 0.10 + 0.30 * raw_sim
                match_level = "KEYWORD"

            # Apply strict role-based penalties
            role = getattr(section, 'law_role', get_section_role(section.title or "", section.simple_explanation))
            if role == "punishment":
                penalty = 0.02
            elif role == "definition":
                penalty = 0.10
            elif role == "procedure":
                penalty = 0.05
            else:
                penalty = 0.0
            final_score = max(0.0, min(final_score - penalty, 1.0))

            scored_candidates.append({
                "section": section,
                "matched_facts": item["matched_facts"],
                "req_all": item["req_all"],
                "req_any": item["req_any"],
                "raw_sim": float(raw_sim),
                "match_level": match_level,
                "penalty": penalty,
                "final_score": final_score,
                "cat_match": sec_cat in all_query_categories,
                "role": role,
                "exp_variant": item["exp_variant"],
                "age_rule": item["age_rule"],
                "elements": elements,
                "status": status
            })

        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        top_score = scored_candidates[0]["final_score"] if scored_candidates else 0.0
        
        accepted_sections = []
        accepted_group_ids = set()

        for item in scored_candidates:
            section = item["section"]
            sec_num = section.section_number
            score = item["final_score"]

            if item["match_level"] == "KEYWORD":
                rejected_laws_list.append({
                    "section": sec_num,
                    "reason": "Keyword similarity alone is insufficient to return this legal provision."
                })
                continue

            min_allowed_score = max(BASE_THRESHOLD, top_score - 0.15)

            if score >= min_allowed_score:
                accepted_sections.append((score, section, item["exp_variant"], item["age_rule"], item["elements"], item["status"]))
                group_id = SECTION_GROUPS.get(sec_num)
                if group_id:
                    accepted_group_ids.add(group_id)

        # Second pass: Accept child/secondary group sections if parent group is accepted
        already_accepted_ids = {s[1].id for s in accepted_sections}
        for item in scored_candidates:
            section = item["section"]
            if section.id in already_accepted_ids or item["match_level"] == "KEYWORD":
                continue
            group_id = SECTION_GROUPS.get(section.section_number)
            if group_id and group_id in accepted_group_ids and item["final_score"] >= BASE_THRESHOLD:
                accepted_sections.append((item["final_score"], section, item["exp_variant"], item["age_rule"], item["elements"], item["status"]))

        # Check if there are no applicable and potential laws to return
        has_applicable = any(s[5] == "strong_match" for s in accepted_sections)
        has_potential = any(s[5] == "potential_match" for s in accepted_sections)
        
        if not has_applicable and not has_potential:
            # Return empty / insufficient-information result
            return LegalRetrievalResult([], incident_summary=generate_incident_summary(facts, language), facts=[], applicable_laws=[], potential_laws=[], rejected_laws=rejected_laws_list, additional_information_needed=[])

        # Group parent-child structure
        grouped_results = []
        seen_groups = {}
        
        applicable_laws_list = []
        potential_laws_list = []

        for score, section, variant, age_rule, elements, status in accepted_sections:
            sec_num = section.section_number
            english_title = getattr(section, "title_en", None) or getattr(section, "title", None) or f"{section.law_name} {section.section_number}"

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

            # Add to applicable/potential lists
            matched_el_list = [el for el, v in elements.items() if v == "SATISFIED"]
            ev_list, reason_str = compile_evidence_and_reason(sec_num, facts, elements)
            
            structured_record = {
                "section": sec_num,
                "title": english_title,
                "status": "strong_match" if status == "strong_match" else "potential_match",
                "matched_elements": matched_el_list,
                "evidence": ev_list,
                "reason": reason_str,
                "source": getattr(section, "source", "Sri Lanka Penal Code"),
                "source_version": getattr(section, "source_version", "1.0.0")
            }

            if status == "strong_match":
                applicable_laws_list.append(structured_record)
            else:
                potential_laws_list.append(structured_record)
                
                # Gather missing facts
                unknown_elements = [el for el, v in elements.items() if v == "UNKNOWN"]
                friendly_questions = {
                    "victim_under_18": "To verify if the victim is a child (under 18).",
                    "offender_has_custody_charge_or_care": "To confirm whether the alleged offender is a parent, guardian, caregiver, or holds custody/care of the child.",
                    "wilful_assault_ill_treatment_neglect_abandonment": "Details about whether the offender's conduct constitutes wilful assault, ill-treatment, neglect, or abandonment.",
                    "conduct_likely_to_cause_suffering_or_injury": "Information about whether the conduct caused or was likely to cause physical/mental suffering or injury to health.",
                    "victim_under_12": "To verify if the victim is under 12 years of age.",
                    "parent_or_person_having_care": "To verify if the offender is a parent or person having care of the child.",
                    "abandonment_or_exposure": "Verification of abandonment or exposure of the child.",
                    "intent_to_wholly_abandon": "Verification of intent to wholly/permanently abandon the child.",
                    "voluntarily_causing_hurt": "Verification of physical assault or intentional actions to cause hurt.",
                    "causing_bodily_pain_disease_infirmity": "Details regarding bodily pain, swelling, bruising, or illness.",
                    "use_of_dangerous_weapon_or_means": "Details about whether a dangerous weapon or dangerous means was used.",
                    "grievous_hurt_category_satisfied": "Information confirming a statutory grievous hurt category.",
                    "sexual_intercourse_penetration": "To verify if sexual penetration or intercourse occurred.",
                    "against_will_or_without_consent_or_under_16": "To verify if the act was against will, without consent, or if the victim was under 16.",
                    "sexual_intercourse_or_grave_abuse": "To verify if sexual intercourse or grave sexual abuse occurred.",
                    "incestuous_relationship": "To confirm if the offender is closely related to the child.",
                    "incestuous_sexual_intercourse": "To verify if incestuous sexual intercourse occurred.",
                    "grave_sexual_conduct": "To verify if grave sexual conduct occurred.",
                    "short_of_intercourse_negation": "To confirm the absence of full sexual intercourse/penetration.",
                    "sexual_harassment_conduct": "Verification of sexual harassment, indecent behavior, or unwelcome comments.",
                    "victim_minor_under_guardianship": "To verify if the victim is under lawful guardianship.",
                    "taking_or_enticing_from_guardian": "Details about whether the child was taken/enticed away from their guardian.",
                    "kidnapping_or_abduction_conduct": "Verification of kidnapping or abduction conduct.",
                    "forced_labour_or_slavery_conduct": "Verification of forced labour, begging, or slavery exploitation.",
                    "victim_under_21": "To verify if the victim is under 21 years of age.",
                    "procuring_conduct": "Verification of procurement for sexual exploitation.",
                    "exploitation_conduct": "Verification of child sexual exploitation or CSAM involvement.",
                    "trafficking_conduct": "Verification of child trafficking conduct.",
                    "adoption_offence_conduct": "Verification of illegal adoption or trafficking conduct.",
                    "soliciting_conduct": "Verification of soliciting a child.",
                    "obscene_material_conduct": "Verification of producing or distributing obscene materials/CSAM.",
                    "computer_or_online_means": "To confirm if online or computer networks were used.",
                    "premises_used_for_abuse": "To confirm if premises were knowingly provided for child abuse.",
                    "begging_conduct": "Verification of whether the child was induced or forced to beg.",
                    "unnatural_carnal_intercourse": "To verify if unnatural carnal intercourse occurred.",
                    "gross_indecency_conduct": "To verify if gross indecency occurred.",
                    "employ_child_as_procurer": "To verify if the child was employed as a procurer.",
                    "traffic_restricted_articles": "To verify if the child was employed to traffic in restricted articles.",
                    "threat_to_cause_alarm_or_force_omission": "To verify if threats were used to cause alarm or force silence/action.",
                    "criminal_intimidation_threat": "To verify if threats of harm or injury were made."
                }
                for el in unknown_elements:
                    friendly = friendly_questions.get(el, f"Verification of element: {el}.")
                    additional_info_list.append({
                        "section": sec_num,
                        "missing_element": el,
                        "reason": friendly
                    })

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

        # Generate structured facts list
        facts_list = []
        for fact_key, val in facts.items():
            if val is not None:
                evidence_text = "Stated in description" if val is True else ("Negated in description" if val is False else "Not stated")
                facts_list.append({
                    "fact": fact_key,
                    "value": val,
                    "evidence": evidence_text
                })

        incident_summary_val = generate_incident_summary(facts, language)

        print(f"FINAL RETURNED LAWS COUNT: {len(grouped_results)}")
        for g in grouped_results:
            print(f"  -> Section {g.section}: {g.title_en} (Variant: {g.explanation_variant}, Age Rule: {g.matched_age_rule}, Score: {g.relevance_score})")

        return LegalRetrievalResult(
            grouped_results,
            incident_summary=incident_summary_val,
            facts=facts_list,
            applicable_laws=applicable_laws_list,
            potential_laws=potential_laws_list,
            rejected_laws=rejected_laws_list,
            additional_information_needed=additional_info_list
        )
    except Exception as e:
        print(f"Legal retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return LegalRetrievalResult([])