import re
from typing import List, Optional, Union

def has_dangerous_weapon_or_means_check(query_lower: str) -> bool:
    english_weapons = [
        "weapon", "weapons", "gun", "pistol", "firearm", "knife", "knives", "blade", "blades", 
        "sword", "swords", "dagger", "axe", "machete", "iron rod", "metal bar", "hammer", "bat", 
        "fire", "poison", "acid", "corrosive", "chemical", "noxious", "explosive", "boiling water", 
        "hot water", "pour hot", "stick", "sticks", "pole", "poles"
    ]
    if any(kw in query_lower for kw in english_weapons):
        return True

    sinhala_weapons = [
        "ආයුධ", "පිහි", "කඩු", "තුවක්කු", "යකඩ පොල්ල", "ගිනි", "ගින්දර", "ගින්න", "ඇසිඩ්", "රසායනික", 
        "පුපුරණ", "පොල්ල", "කෝටු"
    ]
    if any(kw in query_lower for kw in sinhala_weapons):
        return True

    if "විෂ" in query_lower:
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


def extract_victim_age(query: str) -> Optional[int]:
    query_lower = query.lower()
    match_kv = re.search(r"(?:victim\s*age|age)\s*[:=]?\s*(\d{1,2})", query_lower)
    if match_kv:
        return int(match_kv.group(1))

    match_en = re.search(r"(\d{1,2})\s*-\s*years?\s*-\s*old|(\d{1,2})\s*years?\s*old|(\d{1,2})\s*years?\s*of\s*age|aged\s*(\d{1,2})", query_lower)
    if match_en:
        for g in match_en.groups():
            if g:
                return int(g)

    match_si = re.search(r"(?:වයස\s*අවුරුදු|වයස|අවුරුදු)\s*(\d{1,2})", query_lower)
    if match_si:
        return int(match_si.group(1))

    return None


def extract_all_structured_facts(query: str, language: str) -> dict:
    query_lower = query.lower()

    facts = {
        "victim_age": None,
        "victim_sex": None,
        "offender_relationship": None,
        "custody_or_care": None,
        "physical_assault": None,
        "physical_injury": None,
        "injury_severity": None,
        "weapon_or_dangerous_means": None,
        "sexual_contact": None,
        "sexual_act": None,
        "penetration": None,
        "sexual_harassment": None,
        "sexual_image_material": None,
        "online_contact": None,
        "kidnapping": None,
        "taking_from_guardian": None,
        "abduction": None,
        "trafficking": None,
        "commercial_exploitation": None,
        "begging": None,
        "neglect": None,
        "food_deprivation": None,
        "medical_neglect": None,
        "lack_of_supervision": None,
        "abandonment": None,
        "intent_to_wholly_abandon": None,
        "health_suffering": None,
        "threats": None,
        "confinement": None,
        "sexual_touching": None,
        "repeated_conduct": None,
        "adult_offender": None,
        "threat_to_keep_silent": None,
        "threat_of_harm": None,
        "psychological_distress": None,
        "intercourse": None,
        "unnatural_intercourse": None,
        "sodomy": None,
        "gross_indecency": None,
        "employ_child_as_procurer": None,
        "traffic_restricted_articles": None
    }

    # 1. victim_age
    facts["victim_age"] = extract_victim_age(query)

    # 2. victim_sex
    female_kws = ["girl", "daughter", "female", "sister", "she", "her", "ගැහැණු", "දුව", "කාන්තාව"]
    male_kws = ["boy", "son", "male", "brother", "he", "him", "පිරිමි", "පුතා", "මිනිසා"]
    if any(kw in query_lower for kw in female_kws):
        facts["victim_sex"] = "female"
    elif any(kw in query_lower for kw in male_kws):
        facts["victim_sex"] = "male"

    # 3. offender_relationship
    rel_map = {
        "parent": ["father", "mother", "parent", "parents", "dad", "mom", "පියා", "මව", "තාත්තා", "අම්මා", "දෙමාපිය", "දෙමව්පිය"],
        "guardian": ["guardian", "භාරකාර", "භාරකරු", "භාරව සිටින"],
        "caregiver": ["caregiver", "රැකබලා ගන්නා", "රැකවරණය භාර", "භාරව සිටින තැනැත්තා"],
        "teacher": ["teacher", "warden", "nanny", "babysitter", "ගුරු", "ගුරුවරයා", "ගුරුවරිය"],
        "employer": ["employer", "master", "boss", "ස්වාමියා", "හාම්පුතා"],
        "stranger": ["stranger", "unknown person", "නුහුරු", "අමුත්තෙක්"],
        "relative": ["uncle", "aunt", "cousin", "relative", "relatives", "family", "stepfather", "stepmother", "ඥාති", "මාමා", "නැන්දා"]
    }
    for rel, kws in rel_map.items():
        if any(kw in query_lower for kw in kws):
            facts["offender_relationship"] = rel
            break

    # 4. custody_or_care
    custody_pos = ["custody", "charge", "care", "responsible for", "taking care of", "රැකවරණය", "භාරව", "භාරයේ"]
    if facts["offender_relationship"] in ["parent", "guardian", "caregiver"]:
        facts["custody_or_care"] = True
    elif facts["offender_relationship"] == "stranger":
        facts["custody_or_care"] = False
    elif any(kw in query_lower for kw in custody_pos):
        facts["custody_or_care"] = True

    # Clean query for physical checks to avoid false positives from threats
    query_clean_phys = query_lower
    threat_phrases_phys = [
        "threat of harm", "threat of hurt", "threat of injury", "threat of violence",
        "threatened with harm", "threatened with hurt", "threatened with violence",
        "threatened harm", "threatened to harm", "threatened to hurt", "threatened to beat", "threatened to hit",
        "threaten to harm", "threaten to hurt", "threaten to beat", "threaten to hit",
        "හානියක් කරන බවට තර්ජනය", "හානියක් කරන බවට", "පහර දෙන බවට තර්ජනය", "පහර දෙන බවට",
        "මරන බවට තර්ජනය", "මරන බවට", "රිදවන බවට තර්ජනය"
    ]
    for phrase in threat_phrases_phys:
        query_clean_phys = query_clean_phys.replace(phrase, "")

    # 5. physical_assault
    assault_pos = ["hit", "beat", "beaten", "struck", "assault", "assaulted", "slap", "slapped", "punch", "punched", "kick", "kicked", "beating", "පහර", "ගැහුවා", "ගහනවා", "ගහලා", "බැට", "හිංසනය", "තැළුවා", "තලනවා", "stab", "stabbed", "stabbing", "cut", "cutting", "slash", "slashing", "injure", "injures", "injured", "harm", "harmed", "hurt", "ඇන්නා", "ඇනීම", "කැපුවා"]
    assault_neg_regex_en = r"\b(?:no|without|did not|never)\s+(?:physical\s+)?(?:assault|beating|hitting|striking|abuse)\b|\b(?:assault|beating|hitting|striking|abuse)\s+(?:was not|did not|never occurred)\b"
    assault_neg_regex_si = r"(?:පහරදීමක්|පහරදීම්|ගැසීමක්|හිංසනයක්).*?(?:නොවීය|නැත|නොමැත|සිදු වී නැත|සිදු නොවුණි)"
    
    if re.search(assault_neg_regex_en, query_clean_phys) or re.search(assault_neg_regex_si, query_clean_phys):
        facts["physical_assault"] = False
    elif any(kw in query_clean_phys for kw in assault_pos):
        facts["physical_assault"] = True

    # 6. physical_injury
    injury_pos = ["injury", "injured", "wound", "wounded", "bleeding", "fracture", "bruise", "bruises", "pain", "visible injuries", "harm", "swelling", "swollen", "තුවාල", "ලේ ගැලීම", "තැල්ම", "නිල් තැල්ම", "වේදනාව", "ශාරීරික හානි", "ඇඟ රිදෙනවා", "කැක්කුම", "ඉදිමීම", "ඉදිමීම්", "ඉදිමුම්", "තැලීම", "තැලීම්", "තැල්ම", "රිදවීම", "රිදවීම්", "රිදෙව්වා"]
    injury_neg_regex_en = r"\b(?:no|without|free of|did not cause|no bodily)\s+(?:visible\s+)?(?:injury|injuries|bruise|bruises|wound|wounds|swelling|bleeding|pain|harm)\b|\b(?:injury|injuries|bruise|bruises|wound|wounds|swelling|bleeding|pain|harm)\s+(?:were not|did not|was not|not found|not present)\b"
    injury_neg_regex_si = r"(?:තුවාලයක්|තුවාල|හානියක්|වේදනාවක්|රළු ස්පර්ශයක්).*?(?:නොවීය|නැත|නොමැත|සිදු වී නැත|සිදු නොවුණි)"
    
    if re.search(injury_neg_regex_en, query_clean_phys) or re.search(injury_neg_regex_si, query_clean_phys):
        facts["physical_injury"] = False
    elif any(kw in query_clean_phys for kw in injury_pos):
        facts["physical_injury"] = True

    # 7. injury_severity
    grievous_kws = [
        "emasculation", "impotent", "castration", "නපුංසක", "වන්ධ්‍යා",
        "blind", "sight", "deaf", "hearing", "අන්ධ", "පෙනීම", "බිහිරි", "ඇසීම",
        "limb", "joint", "amputation", "amputate", "severed", "අතපය", "අත් පා", "සන්ධි",
        "disfigure", "disfigurement", "scar", "facial", "විකෘති",
        "fracture", "fractured", "dislocate", "dislocated", "bone broken", "broken bone", 
        "broken tooth", "teeth broken", "tooth knocked", "knocked out tooth", 
        "බිඳී", "බිඳීම", "බිඳීම්", "පැනීම", "කැඩී", "හැලී",
        "endanger life", "endangers life", "life-threatening", "critical condition", "icu", "coma", 
        "20 days", "twenty days", "දින 20", "දවස් 20", "මරණාසන්න"
    ]
    if any(kw in query_clean_phys for kw in grievous_kws):
        facts["injury_severity"] = "grievous"
    elif facts["physical_injury"] is True:
        facts["injury_severity"] = "simple"

    # 8. weapon_or_dangerous_means
    weapon_neg_regex_en = r"\b(?:no|without|did not use)\s+(?:dangerous\s+)?(?:weapon|weapons|knife|gun|stick|rod|acid|fire)\b"
    weapon_neg_regex_si = r"(?:ආයුධයක්|ආයුධ|පිහියක්|පොල්ලක්).*?(?:නොවීය|නැත|නොමැත|භාවිතා කළේ නැත)"
    if re.search(weapon_neg_regex_en, query_lower) or re.search(weapon_neg_regex_si, query_lower):
        facts["weapon_or_dangerous_means"] = False
    elif has_dangerous_weapon_or_means_check(query_lower):
        facts["weapon_or_dangerous_means"] = True

    # 9. sexual_contact
    sex_contact_pos = ["touch", "touched", "touching", "private parts", "groped", "indecent touch", "inappropriate touch", "අනුචිත ලෙස ස්පර්ශ", "අනවශ්‍ය ලෙස ස්පර්ශ", "අතපත ගෑම", "ස්පර්ශ කළ", "ස්පර්ශ කිරීම", "ස්පර්ශ", "වැරදි විදියට", "වැරදි ලෙස ස්පර්ශ"]
    sex_contact_neg_regex_en = r"\b(?:no|without|did not)\s+(?:sexual\s+)?(?:touch|touching|private parts)\b"
    sex_contact_neg_regex_si = r"(?:ස්පර්ශයක්|ස්පර්ශ කිරීමක්).*?(?:නොවීය|නැත|නොමැත)"
    if re.search(sex_contact_neg_regex_en, query_lower) or re.search(sex_contact_neg_regex_si, query_lower):
        facts["sexual_contact"] = False
    elif any(kw in query_lower for kw in sex_contact_pos):
        facts["sexual_contact"] = True

    # 10. sexual_act
    sex_act_pos = ["sexual act", "sexual violation", "grave sexual abuse", "sexual acts", "sexual crime", "sexual assault", "ලිංගික ක්‍රියා", "ලිංගික වධදීම", "ලිංගික අපයෝජනය", "අතවරයකට"]
    sex_act_neg_regex_en = r"\b(?:no|without)\s+(?:sexual\s+)?(?:acts?|abuse)\b"
    sex_act_neg_regex_si = r"(?:ලිංගික ක්‍රියාවක්|ලිංගික අපයෝජනයක්).*?(?:නොවීය|නැත|නොමැත)"
    if re.search(sex_act_neg_regex_en, query_lower) or re.search(sex_act_neg_regex_si, query_lower):
        facts["sexual_act"] = False
    elif any(kw in query_lower for kw in sex_act_pos):
        facts["sexual_act"] = True

    # 11. penetration
    penetration_pos = ["rape", "raped", "penetration", "forced intercourse", "forced sex", "penetrated", "statutory rape", "intercourse", "දූෂණය", "ලිංගික සංසර්ගය", "බලහත්කාරයෙන් ලිංගික", "සංසර්ගය"]
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
        facts["penetration"] = False
        facts["intercourse"] = False
    elif any(kw in query_lower for kw in penetration_pos):
        facts["penetration"] = True
        if facts["intercourse"] is None:
            facts["intercourse"] = True

    # 12. sexual_harassment
    harassment_pos = ["sexual harassment", "modesty", "unwelcome sexual", "sexual comments", "catcall", "outrage modesty", "harass", "ලිංගික හිරිහැර", "ලිංගික අතවර", "ලැජ්ජාවට පත්"]
    if any(kw in query_lower for kw in harassment_pos):
        facts["sexual_harassment"] = True

    # 13. sexual_image_material
    image_pos = ["photo", "photos", "video", "videos", "picture", "pictures", "csam", "obscene", "nude", "media", "recording", "publish photo", "ඡායාරූප", "වීඩියෝ", "පින්තූර", "අසභ්‍ය"]
    if any(kw in query_lower for kw in image_pos):
        facts["sexual_image_material"] = True

    # 14. online_contact
    online_pos = ["computer", "internet", "online", "website", "platform", "server", "isp", "service provider", "app", "digital", "social media", "telegram", "whatsapp", "පරිගණක", "අන්තර්ජාලය", "වෙබ්", "ඔන්ලයින්", "සේවා සපයන්නා"]
    if any(kw in query_lower for kw in online_pos):
        facts["online_contact"] = True

    # 15. kidnapping
    kidnap_pos = ["kidnap", "kidnapped", "lawful guardianship", "snatch", "පැහැරගැනීම", "පැහැරගෙන", "භාරකාරත්වයෙන් පැහැර"]
    if any(kw in query_lower for kw in kidnap_pos):
        facts["kidnapping"] = True

    # 16. taking_from_guardian
    taking_pos = ["took the child away", "took away", "enticed away", "enticing from", "භාරකාරත්වයෙන් බැහැර", "රැගෙන ගියා"]
    if any(kw in query_lower for kw in taking_pos):
        facts["taking_from_guardian"] = True

    # 17. abduction
    abduct_pos = ["abduct", "abducted", "forcefully taken", "බලහත්කාරයෙන් රැගෙන", "පැහැරගෙන"]
    if any(kw in query_lower for kw in abduct_pos):
        facts["abduction"] = True

    # 18. trafficking
    trafficking_pos = ["traffic", "trafficking", "sold", "buying", "selling", "transported for exploitation", "human trafficking", "ජාවාරම", "විකිණීම", "ළමා ජාවාරම", "ගනුදෙනු"]
    if any(kw in query_lower for kw in trafficking_pos):
        facts["trafficking"] = True

    # 19. commercial_exploitation
    exploitation_pos = ["procurer", "prostitution", "brothel", "commercial sex", "solicit", "soliciting", "grooming", "pimp", "තැරැව්කාර", "ප්‍රසම්පාදක", "ලිංගික සූරාකෑම"]
    if any(kw in query_lower for kw in exploitation_pos):
        facts["commercial_exploitation"] = True

    # 20. begging
    begging_pos = ["beg", "begging", "alms", "beggar", "සිඟමන්", "හිඟා", "සිඟමන් යැදීම"]
    # Context check: if begging father/mother/parents/caregiver, it is NOT alms begging
    begging_context_en = r"beg(?:ging)?\s+(?:his\s+|her\s+|their\s+)?(?:father|mother|parent|parents|guardian|caregiver|teacher)"
    begging_context_si = r"(?:පියාගෙන්|මවගෙන්|දෙමාපියන්ගෙන්|ඥාතීන්ගෙන්)\s+(?:කෑම\s+)?ඉල්ලා"
    if re.search(begging_context_en, query_lower) or re.search(begging_context_si, query_lower):
        facts["begging"] = False
    elif any(kw in query_lower for kw in begging_pos):
        facts["begging"] = True

    # 21. neglect
    neglect_pos = ["neglect", "without care", "without protection", "no care", "no protection", "not cared for", "not looked after", "නොසලකා", "නොසලකා හැරීම", "ආරක්ෂාව නැහැ", "රැකවරණයක් නැති"]
    neglect_neg = ["no neglect", "proper care", "නොසලකා හැරීමක් නැත"]
    if any(kw in query_lower for kw in neglect_neg):
        facts["neglect"] = False
    elif any(kw in query_lower for kw in neglect_pos):
        facts["neglect"] = True

    # 22. food_deprivation
    food_pos = ["without food", "no food", "starved", "starving", "no food and water", "food deprivation", "nothing to eat", "කෑම නැති", "නිරාහාරව", "කෑම බීම නොදී", "කෑම නොදී", "කෑම ඉල්ලා", "කෑම ඉල්ලයි"]
    if any(kw in query_lower for kw in food_pos):
        facts["food_deprivation"] = True

    # 23. medical_neglect
    medical_pos = ["medical neglect", "no medical", "without medical care", "refused medical treatment", "වෛද්‍ය ප්‍රතිකාර නොදී", "බෙහෙත් නොදී"]
    if any(kw in query_lower for kw in medical_pos):
        facts["medical_neglect"] = True

    # 24. lack_of_supervision
    supervision_pos = ["left alone", "unattended", "unsupervised", "no supervision", "බැලීමට කෙනෙකු නොමැතිව", "තනිවම දමා"]
    if any(kw in query_lower for kw in supervision_pos):
        facts["lack_of_supervision"] = True

    # 25. abandonment
    abandon_pos = ["abandon", "abandoned", "deserted", "left alone in public", "intent to desert", "desertion", "අත්හැර", "අත්හැර දමා", "දමා ගොස්"]
    abandon_neg = ["not abandoned", "did not abandon", "අත්හැර දමා නැත"]
    if any(kw in query_lower for kw in abandon_neg):
        facts["abandonment"] = False
    elif any(kw in query_lower for kw in abandon_pos):
        facts["abandonment"] = True

    # 26. intent_to_wholly_abandon
    intent_abandon_pos = ["wholly abandon", "intent to wholly", "intent to desert", "permanently left", "සම්පූර්ණයෙන්ම අත්හැර"]
    if any(kw in query_lower for kw in intent_abandon_pos):
        facts["intent_to_wholly_abandon"] = True
    elif facts["abandonment"] is True and facts["custody_or_care"] is True:
        facts["intent_to_wholly_abandon"] = True

    # 27. health_suffering
    suffering_pos = ["causing suffering", "cause suffering", "injury to health", "suffering to health", "හානියක් සිදුකිරීම", "පීඩාවක් ඇතිකිරීම", "හඬා", "හඬමින්", "වැලපෙමින්", "crying", "weeping"]
    if any(kw in query_lower for kw in suffering_pos):
        facts["health_suffering"] = True

    # 28. threats
    threat_pos = ["threatened", "warned not to tell", "frightened into silence", "threat", "scare", "scared", "fear", "don't tell", "afraid", "intimidated", "silenced", "තර්ජනය", "කිසිවෙකුට නොකියන ලෙස", "නොකියන ලෙස", "බිය ගැන්වූ", "කියන්න එපා කියා"]
    if any(kw in query_lower for kw in threat_pos):
        facts["threats"] = True

    # 29. confinement
    confinement_pos = ["confinement", "confined", "locked inside", "locked in a room", "imprisoned", "wahuwa", "hira", "හිරකර", "කොටු කර", "වසා තිබූ"]
    if any(kw in query_lower for kw in confinement_pos):
        facts["confinement"] = True

    # 30. sexual_touching
    sexual_touching_pos = ["touch", "touched", "touching", "private parts", "groped", "indecent touch", "inappropriate touch", "අනුචිත ලෙස ස්පර්ශ", "අනවශ්‍ය ලෙස ස්පර්ශ", "අතපත ගෑම", "ස්පර්ශ කළ", "ස්පර්ශ කිරීම", "ස්පර්ශ", "වැරදි විදියට", "වැරදි ලෙස ස්පර්ශ"]
    sexual_touching_neg_regex_en = r"\b(?:no|without|did not)\s+(?:sexual\s+)?(?:touch|touching|private parts)\b"
    sexual_touching_neg_regex_si = r"(?:ස්පර්ශයක්|ස්පර්ශ කිරීමක්).*?(?:නොවීය|නැත|නොමැත)"
    if re.search(sexual_touching_neg_regex_en, query_lower) or re.search(sexual_touching_neg_regex_si, query_lower):
        facts["sexual_touching"] = False
    elif any(kw in query_lower for kw in sexual_touching_pos):
        facts["sexual_touching"] = True
    elif facts["sexual_contact"] is True:
        facts["sexual_touching"] = True

    # 31. repeated_conduct
    repeated_pos = ["repeated", "repeatedly", "multiple times", "often", "frequently", "ongoing", "continuous", "over and over", "several times", "නැවත නැවතත්", "නිරන්තරයෙන්", "පිට පිට", "බොහෝ වාරයක්", "නිරතුරුවම"]
    if any(kw in query_lower for kw in repeated_pos):
        facts["repeated_conduct"] = True

    # 32. adult_offender
    adult_pos = ["adult", "grown-up", "man", "woman", "uncle", "aunt", "parent", "caregiver", "guardian", "known adult", "adult offender", "වැඩිහිටි", "වැඩිහිටියෙකු", "වැඩිහිටියා"]
    if any(kw in query_lower for kw in adult_pos):
        facts["adult_offender"] = True
    elif facts["offender_relationship"] in ["parent", "guardian", "caregiver", "teacher", "employer", "relative"]:
        facts["adult_offender"] = True

    # 33. threat_to_keep_silent
    silent_pos = ["keep silent", "keep quiet", "dont tell", "don't tell", "not to tell", "warned not to tell", "threatened to keep silent", "silence", "tells anyone", "if you tell", "secret", "keep it secret", "නිශ්ශබ්දව", "නොකියන ලෙස", "නොකියන ලෙසට", "කියන්න එපා", "කිසිවෙකුට නොකියන"]
    if any(kw in query_lower for kw in silent_pos):
        facts["threat_to_keep_silent"] = True

    # 34. threat_of_harm
    harm_threat_pos = ["threat of harm", "threatened with harm", "threaten to harm", "threaten to beat", "kill", "hurt", "threat of violence", "threaten to injure", "harm if", "හානියක් කරන", "හානි කරන", "පහර දෙන බවට තර්ජනය", "මරණ තර්ජන", "මරන බවට", "මරනවා", "තර්ජනය"]
    if any(kw in query_lower for kw in harm_threat_pos):
        facts["threat_of_harm"] = True

    # 35. psychological_distress
    distress_pos = ["distress", "fear", "scared", "afraid", "terrified", "traumatized", "depression", "anxiety", "psychological", "emotional pain", "mental suffering", "trauma", "බිය", "බය", "මානසික පීඩාව", "මානසික කෲරත්වය", "මානසික පීඩා"]
    if any(kw in query_lower for kw in distress_pos):
        facts["psychological_distress"] = True

    # 36. intercourse
    if facts["intercourse"] is None:
        intercourse_pos = ["intercourse", "sex", "sexual intercourse", "සංසර්ගය", "ලිංගික සංසර්ගය"]
        intercourse_neg_regex_en = r"\b(?:no|without|did not|never)\s+(?:explicit\s+|sexual\s+)?(?:intercourse)\b|\b(?:intercourse)\s+(?:was not|did not|never occurred|is absent)\b"
        intercourse_neg_regex_si = r"(?:සංසර්ගයක්|සංසර්ගය).*?(?:නොවීය|නැත|නොමැත|සිදු වී නැත|සිදු නොවුණි)"
        if re.search(intercourse_neg_regex_en, query_lower) or re.search(intercourse_neg_regex_si, query_lower):
            facts["intercourse"] = False
        elif any(kw in query_lower for kw in intercourse_pos):
            facts["intercourse"] = True

    # 37. unnatural_intercourse
    unnatural_intercourse_pos = ["unnatural carnal", "buggery", "against the order of nature", "anal sex", "oral sex", "sodomy", "ස්වභාවධර්මයට පටහැනි", "ගුද සංසර්ගය", "මුඛ සංසර්ගය", "අස්වාභාවික ලිංගික"]
    if any(kw in query_lower for kw in unnatural_intercourse_pos):
        facts["unnatural_intercourse"] = True

    # 38. sodomy
    sodomy_pos = ["sodomy", "sodomized", "buggery", "anal sex", "ගුද සංසර්ගය"]
    if any(kw in query_lower for kw in sodomy_pos):
        facts["sodomy"] = True

    # 39. gross_indecency
    gross_indecency_pos = ["gross indecency", "grossly indecent", "gross indecency act", "බරපතල අශෝභන ක්‍රියා", "අශෝභන ක්‍රියා"]
    if any(kw in query_lower for kw in gross_indecency_pos):
        facts["gross_indecency"] = True

    # 40. employ_child_as_procurer
    procurer_pos = ["employing children to act as procurers", "hiring children to act as procurers", "employ a child as a procurer", "hire a child as a procurer", "act as a procurer", "procurer", "තැරැව්කරුවන් ලෙස ළමයින් යොදා ගැනීම", "තැරැව්කරුවන්"]
    if any(kw in query_lower for kw in procurer_pos) or (facts["commercial_exploitation"] is True and "procur" in query_lower):
        facts["employ_child_as_procurer"] = True

    # 41. traffic_restricted_articles
    traffic_articles_pos = ["traffic in restricted articles", "trafficking restricted articles", "restricted articles", "sell drugs", "sell liquor", "තහනම් භාණ්ඩ ජාවාරම", "තහනම් ද්‍රව්‍ය", "මත්ද්‍රව්‍ය"]
    if any(kw in query_lower for kw in traffic_articles_pos):
        facts["traffic_restricted_articles"] = True

    # Fill in secondary inferences
    if facts["food_deprivation"] is True or facts["medical_neglect"] is True or facts["lack_of_supervision"] is True:
        if facts["neglect"] is None or facts["neglect"] is False:
            facts["neglect"] = True

    if facts["physical_assault"] is True:
        if facts["health_suffering"] is None:
            facts["health_suffering"] = True

    return facts


def extract_canonical_facts(query: str, language: str = None) -> List[str]:
    facts_dict = extract_all_structured_facts(query, language or "en")
    extracted_canonical = []
    
    if facts_dict["physical_assault"] is True:
        extracted_canonical.append("physical_assault")
    if facts_dict["physical_injury"] is True:
        extracted_canonical.append("physical_injury")
    if facts_dict["weapon_or_dangerous_means"] is True:
        extracted_canonical.append("restricted_articles")
    if facts_dict["sexual_contact"] is True or facts_dict["sexual_touching"] is True:
        extracted_canonical.append("sexual_contact")
    if facts_dict["sexual_act"] is True:
        extracted_canonical.append("sexual_act")
    if facts_dict["penetration"] is True or facts_dict["intercourse"] is True:
        extracted_canonical.append("penetration")
    if facts_dict["sexual_harassment"] is True:
        extracted_canonical.append("sexual_harassment")
    if facts_dict["sexual_image_material"] is True:
        extracted_canonical.append("sexual_image_material")
    if facts_dict["online_contact"] is True:
        extracted_canonical.append("online_contact")
    if facts_dict["kidnapping"] is True or facts_dict["abduction"] is True:
        extracted_canonical.append("kidnapping")
    if facts_dict["trafficking"] is True:
        extracted_canonical.append("trafficking")
    if facts_dict["commercial_exploitation"] is True:
        extracted_canonical.append("commercial_exploitation")
    if facts_dict["begging"] is True:
        extracted_canonical.append("begging")
    if facts_dict["neglect"] is True:
        extracted_canonical.append("neglect")
    if facts_dict["abandonment"] is True:
        extracted_canonical.append("abandonment")
    if facts_dict["threats"] is True or facts_dict["threat_of_harm"] is True or facts_dict["threat_to_keep_silent"] is True:
        extracted_canonical.append("threats")

    return sorted(list(set(extracted_canonical)))
