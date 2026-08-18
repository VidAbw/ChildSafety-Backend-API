from typing import List, Set

CANONICAL_FACT_MAPPINGS = {
    "child_victim": {
        "en": ["child", "minor", "boy", "girl", "kid", "toddler", "infant", "under 18", "10-year-old", "student", "son", "daughter", "schoolchild"],
        "si": ["දරුවා", "දරුවෙකු", "ළමයා", "ළමයෙකු", "කුඩා දරුවා", "බාලවයස්කාර", "පොඩි එක්කෙනා", "පුතා", "දුව"]
    },
    "physical_assault": {
        "en": ["hit", "beat", "beaten", "struck", "assault", "assaulted", "slap", "slapped", "punch", "punched", "kick", "kicked", "iron rod", "stick", "beating", "physically harmed", "harm"],
        "si": ["පහර", "ගැහුවා", "පහර දුන්නා", "පහරදීම", "වධ හිංසා", "වධහිංසා", "බැට දුන්නා", "ගහනවා", "කෝටුවෙන්", "යකඩ පොල්ලෙන්", "ශාරීරික හානි"]
    },
    "physical_injury": {
        "en": ["injury", "injured", "wound", "wounded", "bleeding", "fracture", "bruise", "bruises", "pain", "visible injuries", "harm"],
        "si": ["තුවාල", "ලේ ගැලීම", "තැල්ම", "නිල් තැල්ම", "වේදනාව", "ශාරීරික හානි", "තුවාල ඇති"]
    },
    "cruelty": {
        "en": ["cruelty", "ill-treat", "ill-treatment", "suffering", "torture", "corporal punishment", "cruel", "willfully assault"],
        "si": ["කෲර", "කෲර ලෙස", "හිංසා", "වධදීම", "හිංසා කිරීම", "වධ හිංසා"]
    },
    "sexual_contact": {
        "en": [
            "inappropriate touching", "touched inappropriately", "sexual touching", "unwanted touching", 
            "touch", "touched", "touching", "private parts", "groped", "indecent touch", "inappropriate touch", "strip", "stripped", "undressed"
        ],
        "si": [
            "අනුචිත ලෙස ස්පර්ශ", "අනවශ්ය ලෙස ස්පර්ශ", "අනවශ්‍ය ලෙස ස්පර්ශ", "අසභ්ය ලෙස ස්පර්ශ", 
            "අසභ්‍ය ලෙස ස්පර්ශ", "ලිංගික ලෙස ස්පර්ශ", "අතපත ගෑම", "ස්පර්ශ කළ", "ස්පර්ශ කිරීම", "අසභ්‍ය ලෙස", "ස්පර්ශ",
            "ඇඳුම් ගලවා", "ඇඳුම් ගැලවීම"
        ]
    },
    "sexual_act": {
        "en": [
            "sexual abuse", "sexual act", "sexual violation", "grave sexual abuse", "unnatural act", 
            "sexual acts", "sexual crime", "sexual assault"
        ],
        "si": [
            "ලිංගික අපයෝජනය", "ලිංගික ක්‍රියා", "ලිංගික වධදීම", "ගුරුතර ලිංගික අපයෝජනය", "ලිංගික අතවර", "ලිංගික", "අතවරයකට", "අතවර", "ඇඳුම් ගලවා"
        ]
    },
    "penetration": {
        "en": ["rape", "raped", "penetration", "forced intercourse", "forced sex", "penetrated", "statutory rape", "intercourse"],
        "si": ["දූෂණය", "දූෂණය කර", "ලිංගික සංසර්ගය", "බලහත්කාරයෙන් ලිංගික", "සංසර්ගය"]
    },
    "incest_relation": {
        "en": ["father", "dad", "mother", "mom", "brother", "sister", "uncle", "aunt", "relative", "family", "stepfather", "stepmother", "cousin", "grandfather"],
        "si": ["පියා", "තාත්තා", "මව", "අම්මා", "සහෝදරයා", "සහෝදරිය", "මාමා", "ඥාතියා", "ඥාතියෙකු", "පවුලේ", "ලෙයින්"]
    },
    "sexual_harassment": {
        "en": ["sexual harassment", "modesty", "unwelcome sexual", "sexual comments", "catcall", "outrage modesty", "harass"],
        "si": ["ලිංගික හිරිහැර", "ලිංගික අතවර", "අතවරයකට", "අතවර", "ලැජ්ජාවට පත්"]
    },
    "threat_intimidation": {
        "en": [
            "threatened", "warned not to tell", "frightened into silence", "threat", "scare", 
            "scared", "fear", "don't tell", "afraid", "intimidated", "silenced"
        ],
        "si": [
            "තර්ජනය", "කිසිවෙකුට නොකියන ලෙස", "නොකියන ලෙස", "බිය ගැන්වූ", "කියන්න එපා කියා", 
            "බිය", "බියෙන්", "තර්ජනය කර", "බියවැද්දීම"
        ]
    },
    "neglect": {
        "en": ["neglect", "left alone", "without food", "without care", "unattended", "without protection", "starve", "starving", "no food", "no care"],
        "si": ["නොසලකා", "නොසලකා හැරීම", "කෑම නැහැ", "ආරක්ෂාව නැහැ", "තනිව", "රැකවරණයක් නැති", "කන්න බොන්න"]
    },
    "abandonment": {
        "en": ["abandon", "abandoned", "deserted", "left alone in public", "intent to desert", "desertion"],
        "si": ["අත්හැර", "අත්හැර දමා", "දමා ගොස්"]
    },
    "forced_labour": {
        "en": ["forced labour", "forced labor", "child soldier", "armed conflict", "recrypted for war", "forced work"],
        "si": ["බලහත්කාර ශ්‍රමය", "සන්නද්ධ ගැටුම්", "හමුදාවට", "බලහත්කාර"]
    },
    "debt_bondage": {
        "en": ["debt bondage", "serfdom", "bound for debt", "pawned child"],
        "si": ["ණය ගැති", "ණය ගැති භාවය"]
    },
    "slavery": {
        "en": ["slavery", "slave", "enslaved", "bought child"],
        "si": ["වහල්භාවය", "වහල්", "වහල් සේවය"]
    },
    "kidnapping": {
        "en": ["kidnap", "kidnapped", "abduct", "abducted", "entice", "taken away", "lawful guardianship", "snatch"],
        "si": ["පැහැරගැනීම", "පැහැරගෙන", "භාරකාරත්වයෙන් පැහැර", "භාරකාරත්වයෙන්"]
    },
    "trafficking": {
        "en": ["traffic", "trafficking", "sold", "buying", "selling", "transported for exploitation", "human trafficking"],
        "si": ["ජාවාරම", "විකිණීම", "ළමා ජාවාරම", "ගනුදෙනු"]
    },
    "commercial_exploitation": {
        "en": ["procurer", "prostitution", "brothel", "commercial sex", "solicit", "grooming", "pimp"],
        "si": ["තැරැව්කාර", "ප්‍රසම්පාදක", "ලිංගික සූරාකෑම"]
    },
    "online_contact": {
        "en": ["computer", "internet", "online", "website", "platform", "server", "isp", "service provider", "app", "digital", "social media", "telegram", "whatsapp"],
        "si": ["පරිගණක", "අන්තර්ජාලය", "වෙබ්", "ඔන්ලයින්", "සේවා සපයන්නා"]
    },
    "sexual_image_material": {
        "en": ["photo", "photos", "video", "videos", "picture", "pictures", "csam", "obscene", "nude", "media", "recording", "publish photo"],
        "si": ["ඡායාරූප", "වීඩියෝ", "පින්තූර", "අසභ්‍ය"]
    },
    "restricted_articles": {
        "en": ["drug", "drugs", "narcotics", "contraband", "weapons", "restricted articles", "mule", "illegal drugs"],
        "si": ["මත්ද්‍රව්‍ය", "ජාවාරම්", "තහනම් ද්‍රව්‍ය", "ආයුධ"]
    },
    "adoption_offence": {
        "en": ["adopt", "adoption", "illegal adoption"],
        "si": ["දරුකමට", "දරුකමට ගැනීම"]
    },
    "begging": {
        "en": ["beg", "begging", "alms", "beggar"],
        "si": ["සිඟමන්", "හිඟා", "සිඟමන් යැදීම"]
    }
}


def extract_canonical_facts(query: str, language: str = None) -> List[str]:
    """
    Extracts canonical fact identifiers from query text in English or Sinhala.
    """
    query_lower = query.lower()
    extracted_facts = set()

    # Define negation phrases that should suppress penetration fact
    penetration_negations = [
        "without intercourse",
        "no intercourse",
        "without penetration",
        "no penetration",
        "did not penetrate",
        "no sexual intercourse",
    ]

    for fact_id, lang_dict in CANONICAL_FACT_MAPPINGS.items():
        # Check English phrases
        for phrase in lang_dict.get("en", []):
            if phrase in query_lower:
                extracted_facts.add(fact_id)
                break
        
        # Check Sinhala phrases
        if fact_id not in extracted_facts:
            for phrase in lang_dict.get("si", []):
                if phrase in query_lower:
                    extracted_facts.add(fact_id)
                    break

    # Remove penetration fact if any negation phrase is present
    if "penetration" in extracted_facts:
        if any(neg in query_lower for neg in penetration_negations):
            extracted_facts.remove("penetration")

    # Ensure online_contact is only added when relevant keywords exist
    if "online_contact" in extracted_facts:
        online_keywords = set(CANONICAL_FACT_MAPPINGS.get("online_contact", {}).get("en", []))
        online_keywords.update(CANONICAL_FACT_MAPPINGS.get("online_contact", {}).get("si", []))
        if not any(kw in query_lower for kw in online_keywords):
            extracted_facts.remove("online_contact")

    return sorted(list(extracted_facts))


import re
from typing import Optional

def extract_victim_age(query: str) -> Optional[int]:
    """
    Extracts numerical age of victim from query string if specified.
    Matches English & Sinhala age patterns like:
    - "Victim age = 25", "age: 25", "age 25", "25 years old", "25-year-old", "25 years of age", "aged 25"
    - "වයස අවුරුදු 25", "වයස 25", "අවුරුදු 25"
    """
    query_lower = query.lower()
    
    # 1. Explicit key-value pattern (e.g. "victim age = 25", "age = 25", "age: 25", "victim age 25")
    match_kv = re.search(r"(?:victim\s*age|age)\s*[:=]?\s*(\d{1,2})", query_lower)
    if match_kv:
        return int(match_kv.group(1))

    # 2. English patterns: "25-year-old", "25 year old", "25 years old", "25 years of age", "aged 25"
    match_en = re.search(r"(\d{1,2})\s*-\s*years?\s*-\s*old|(\d{1,2})\s*years?\s*old|(\d{1,2})\s*years?\s*of\s*age|aged\s*(\d{1,2})", query_lower)
    if match_en:
        for g in match_en.groups():
            if g:
                return int(g)

    # 3. Sinhala patterns: "වයස අවුරුදු 25", "වයස 25", "අවුරුදු 25"
    match_si = re.search(r"(?:වයස\s*අවුරුදු|වයස|අවුරුදු)\s*(\d{1,2})", query_lower)
    if match_si:
        return int(match_si.group(1))

    return None
