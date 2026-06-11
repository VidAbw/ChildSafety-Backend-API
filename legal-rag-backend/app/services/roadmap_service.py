from typing import List, Dict

# Pool of all possible roadmap steps with bilingual text and icons
ROADMAP_STEPS_POOL = {
    "immediate_safety": {
        "title_en": "Ensure Immediate Safety",
        "title_si": "වහාම ආරක්ෂාව තහවුරු කරන්න",
        "description_en": "Remove the child from the immediate environment where the abuse took place and relocate them to a safe, trusted space.",
        "description_si": "අපයෝජනය සිදුවූ පරිසරයෙන් දරුවා වහාම ඉවත් කර ආරක්ෂිත, විශ්වාසදායක ස්ථානයකට රැගෙන යන්න.",
        "icon": "shield"
    },
    "evidence_preservation": {
        "title_en": "Preserve Forensic Evidence",
        "title_si": "විද්‍යාත්මක සාක්ෂි සුරකින්න",
        "description_en": "Do not wash the child, change their clothes, or clean any surfaces/items related to the incident before a medico-legal examination.",
        "description_si": "වෛද්‍ය පරීක්ෂණයට පෙර දරුවා සේදීම, ඇඳුම් මාරු කිරීම හෝ සිද්ධියට අදාළ ද්‍රව්‍ය පිරිසිදු කිරීමෙන් වළකින්න.",
        "icon": "inventory"
    },
    "medical_care": {
        "title_en": "Seek Urgent Medical Attention",
        "title_si": "හදිසි වෛද්‍ය ප්‍රතිකාර ලබාගන්න",
        "description_en": "Take the child to the nearest government hospital. Ask to see a Judicial Medical Officer (JMO) for a formal medico-legal exam.",
        "description_si": "දරුවා ළඟම ඇති රජයේ රෝහලට රැගෙන යන්න. වෛද්‍ය පරීක්ෂණයක් සඳහා අධිකරණ වෛද්‍ය නිලධාරී (JMO) හමුවීමට ඉල්ලන්න.",
        "icon": "local_hospital"
    },
    "police_report": {
        "title_en": "Report to the Police",
        "title_si": "පොලිසියට වාර්තා කරන්න",
        "description_en": "File a report immediately by calling the Police Emergency Hotline (119) or visiting the nearest police station.",
        "description_si": "පොලිස් හදිසි ඇමතුම් අංකය (119) ඇමතීමෙන් හෝ ළඟම ඇති පොලිස් ස්ථානයට යාමෙන් වහාම පැමිණිල්ලක් ඉදිරිපත් කරන්න.",
        "icon": "gavel"
    },
    "ncpa_referral": {
        "title_en": "Contact NCPA Helpline",
        "title_si": "ජාතික ළමා ආරක්ෂක අධිකාරිය අමතන්න",
        "description_en": "Report the incident to the National Child Protection Authority (NCPA) hotline at 1929 for protection and legal aid.",
        "description_si": "දරුවාට ආරක්ෂාව සහ නීතිමය සහාය ලබා ගැනීම සඳහා ජාතික ළමා ආරක්ෂක අධිකාරියේ (NCPA) 1929 ක්ෂණික ඇමතුම් අංකයට වාර්තා කරන්න.",
        "icon": "phone"
    },
    "psychological_support": {
        "title_en": "Provide Trauma Counseling",
        "title_si": "මානසික සහාය සහ උපදෙස් ලබා දෙන්න",
        "description_en": "Connect the child with a certified child psychologist or counselor to address trauma, anxiety, or emotional distress.",
        "description_si": "කනස්සල්ල, බිය හෝ මානසික කෲරත්වය සමනය කිරීම සඳහා දරුවා සහතිකලත් ළමා මනෝ විද්‍යාඥයෙකු හෝ උපදේශකයෙකු වෙත යොමු කරන්න.",
        "icon": "spa"
    },
    "safe_shelter": {
        "title_en": "Arrange Safe Shelter",
        "title_si": "ආරක්ෂිත නවාතැන් සලසා දෙන්න",
        "description_en": "If the home environment is unsafe, coordinate with NCPA or social services to arrange temporary safe shelter placement.",
        "description_si": "නිවාස පරිසරය ආරක්ෂිත නොවේ නම්, තාවකාලික ආරක්ෂිත නවාතැනක් සඳහා NCPA හෝ සමාජ සේවා නිලධාරීන් සමඟ සම්බන්ධීකරණය කරන්න.",
        "icon": "home"
    },
    "follow_up_monitoring": {
        "title_en": "Initiate Welfare Monitoring",
        "title_si": "සුභසාධන නිරීක්ෂණ ආරම්භ කරන්න",
        "description_en": "Engage a local Child Rights Promotion Officer (CRPO) to monitor the child's rehabilitation, school environment, and safety.",
        "description_si": "දරුවාගේ පුනරුත්ථාපනය, පාසල් පරිසරය සහ ආරක්ෂාව නිරීක්ෂණය කිරීම සඳහා ප්‍රාදේශීය ළමා හිමිකම් ප්‍රවර්ධන නිලධාරී (CRPO) සම්බන්ධ කරගන්න.",
        "icon": "visibility"
    }
}

# Base roadmap step templates by category
ROADMAP_TEMPLATES = {
    "sexual_abuse": ["immediate_safety", "evidence_preservation", "medical_care", "police_report", "ncpa_referral", "psychological_support"],
    "physical_abuse": ["immediate_safety", "medical_care", "police_report", "ncpa_referral", "follow_up_monitoring"],
    "neglect": ["immediate_safety", "medical_care", "ncpa_referral", "safe_shelter", "follow_up_monitoring"],
    "emotional_abuse": ["immediate_safety", "psychological_support", "ncpa_referral", "follow_up_monitoring"],
    "trafficking_exploitation": ["immediate_safety", "police_report", "ncpa_referral", "safe_shelter", "psychological_support"],
    "psychological_trauma_counseling_need": ["immediate_safety", "psychological_support", "follow_up_monitoring"],
    "general_child_protection": ["immediate_safety", "ncpa_referral", "follow_up_monitoring"]
}

def detect_flags(description: str, category: str) -> Dict[str, bool]:
    """
    Analyzes the user's description (English/Sinhala) and category to set risk/support flags.
    """
    desc_lower = description.lower()
    
    # English keywords
    danger_keywords_en = ["danger", "kill", "threat", "weapon", "knife", "gun", "attacking", "now", "captive", "locked", "choking", "strangled", "run away", "escape"]
    medical_keywords_en = ["bleed", "bleeding", "broken", "fracture", "cut", "wound", "injured", "hospital", "pain", "unconscious", "pregnancy", "pregnant", "doctor", "clinic"]
    evidence_keywords_en = ["recent", "today", "yesterday", "last night", "hours ago", "just now", "now", "semen", "clothing", "stained"]
    psychological_keywords_en = ["trauma", "depressed", "fear", "scared", "anxiety", "crying", "cried", "sad", "nightmare", "suicidal", "mental", "counseling", "counselor", "therapy"]
    shelter_keywords_en = ["shelter", "home", "no place", "nowhere to stay", "homeless", "kicked out", "runaway", "abandoned"]
    monitoring_keywords_en = ["neglect", "ongoing", "monitoring", "welfare", "school", "regular", "visit", "repeated", "always"]

    # Sinhala keywords
    danger_keywords_si = ["අනතුර", "මරන්න", "තර්ජන", "ආයුධ", "පිහිය", "තුවක්කු", "පහරදෙනවා", "දැන්", "හිරකර", "පැනලා"]
    medical_keywords_si = ["ලේ", "රුධිරය", "තුවාල", "රෝහල", "අමාරුයි", "සිහිය නැති", "ගැබ්ගෙන", "වෛද්‍ය"]
    evidence_keywords_si = ["අද", "ඊයේ", "දැන්", "පෙරේදා"]
    psychological_keywords_si = ["කනස්සල්ල", "මානසික", "බය", "අඬනවා", "කඳුළු", "කාංසාව", "සිහින"]
    shelter_keywords_si = ["නවාතැන්", "නිවසක් නැහැ", "අත්හැර දැමූ", "පාරේ"]
    monitoring_keywords_si = ["නිරන්තර", "නිරීක්ෂණය", "නිතරම", "පාසල්"]

    # Combine lists
    danger_words = danger_keywords_en + danger_keywords_si
    medical_words = medical_keywords_en + medical_keywords_si
    evidence_words = evidence_keywords_en + evidence_keywords_si
    psychological_words = psychological_keywords_en + psychological_keywords_si
    shelter_words = shelter_keywords_en + shelter_keywords_si
    monitoring_words = monitoring_keywords_en + monitoring_keywords_si

    # Flag checks
    immediate_danger = any(w in desc_lower for w in danger_words)
    medical_urgency = any(w in desc_lower for w in medical_words)
    
    # Evidence preservation is needed if recent keywords are found, especially in physical/sexual abuse
    has_recent_keyword = any(w in desc_lower for w in evidence_words)
    needs_evidence_preservation = has_recent_keyword or (category in ["sexual_abuse", "physical_abuse"] and "recent" in desc_lower)
    
    needs_police_report = immediate_danger or category in ["sexual_abuse", "physical_abuse", "trafficking_exploitation"]
    needs_ncpa_referral = True  # NCPA is always referred for child protection cases in Sri Lanka
    needs_psychological_support = any(w in desc_lower for w in psychological_words) or category in ["emotional_abuse", "psychological_trauma_counseling_need"]
    needs_safe_shelter = any(w in desc_lower for w in shelter_words) or category in ["trafficking_exploitation"] or (category == "neglect" and ("abandoned" in desc_lower or "පාරේ" in desc_lower))
    needs_follow_up_monitoring = any(w in desc_lower for w in monitoring_words) or category in ["neglect", "emotional_abuse"]

    return {
        "immediate_danger": immediate_danger,
        "medical_urgency": medical_urgency,
        "needs_evidence_preservation": needs_evidence_preservation,
        "needs_police_report": needs_police_report,
        "needs_ncpa_referral": needs_ncpa_referral,
        "needs_psychological_support": needs_psychological_support,
        "needs_safe_shelter": needs_safe_shelter,
        "needs_follow_up_monitoring": needs_follow_up_monitoring,
    }

def generate_roadmap(description: str, abuse_category: str, language: str = "en") -> List[str]:
    """
    Generates a dynamic, prioritized decision roadmap based on category and description flags.
    Returns a list of formatted strings.
    """
    # 1. Detect flags based on case description and category
    flags = detect_flags(description, abuse_category)
    
    # 2. Select base steps for category
    base_steps = list(ROADMAP_TEMPLATES.get(abuse_category, ROADMAP_TEMPLATES["general_child_protection"]))
    steps_list = list(base_steps)
    
    # 3. Insert any missing steps that are flagged
    if flags["medical_urgency"] and "medical_care" not in steps_list:
        steps_list.append("medical_care")
    if flags["needs_evidence_preservation"] and "evidence_preservation" not in steps_list:
        steps_list.append("evidence_preservation")
    if (flags["needs_police_report"] or flags["immediate_danger"]) and "police_report" not in steps_list:
        steps_list.append("police_report")
    if flags["needs_psychological_support"] and "psychological_support" not in steps_list:
        steps_list.append("psychological_support")
    if flags["needs_safe_shelter"] and "safe_shelter" not in steps_list:
        steps_list.append("safe_shelter")
    if flags["needs_follow_up_monitoring"] and "follow_up_monitoring" not in steps_list:
        steps_list.append("follow_up_monitoring")
        
    # 4. Set priorities for all steps (lower numbers sort first)
    priorities = {
        "immediate_safety": 10,
        "evidence_preservation": 20,
        "medical_care": 30,
        "police_report": 40,
        "ncpa_referral": 50,
        "safe_shelter": 60,
        "psychological_support": 70,
        "follow_up_monitoring": 80
    }
    
    # Re-order dynamically based on flags
    if flags["medical_urgency"]:
        priorities["medical_care"] = 15  # Bring medical care early
    if flags["needs_evidence_preservation"]:
        priorities["evidence_preservation"] = 12  # Must happen before JMO medical exam / washing
    if flags["immediate_danger"]:
        priorities["police_report"] = 18  # Report to police early if danger exists
        
    # Sort the steps list by their dynamic priority
    sorted_steps = sorted(steps_list, key=lambda s: priorities.get(s, 99))
    
    # 5. Populate and format steps response (formatted strings)
    final_roadmap = []
    for index, step_id in enumerate(sorted_steps, 1):
        step_data = ROADMAP_STEPS_POOL.get(step_id)
        if step_data:
            if language == "si":
                title = step_data["title_si"]
                desc = step_data["description_si"]
            else:
                title = step_data["title_en"]
                desc = step_data["description_en"]
            # Format as: "1. Title: Description"
            final_roadmap.append(f"{index}. {title}: {desc}")
            
    return final_roadmap
