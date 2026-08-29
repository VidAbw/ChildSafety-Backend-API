import re
from typing import List, Optional, Union

def match_keyword(kw: str, text: str) -> bool:
    """
    Checks if a keyword is present in text, using word boundaries for alphabetic
    and Sinhala keywords to avoid false positives (e.g. 'he' in 'there').
    """
    kw = kw.strip().lower()
    text = text.lower()
    if not kw:
        return False
        
    # Check if keyword contains alphanumeric characters
    if kw.isalnum() or all(c.isalnum() or c.isspace() or c == '-' for c in kw):
        # English or alphanumeric word boundary check
        if any(c.isalpha() for c in kw):
            return bool(re.search(rf"\b{re.escape(kw)}\b", text))
        else:
            # Sinhala boundary check using spaces and punctuation
            return bool(re.search(rf"(?:^|\s|[.,!?;:\-()\"'{{\}}[\]])" + re.escape(kw) + rf"(?:$|\s|[.,!?;:\-()\"'{{\}}[\]])", text))
    return kw in text


def check_fact_presence_and_negation(query_lower: str, positive_kws: List[str], negation_patterns: List[str]) -> Optional[bool]:
    """
    Determines if a fact is True (present and not negated), False (explicitly negated),
    or None (not mentioned/Unknown).
    """
    # 1. Check negation patterns using regex
    for pat in negation_patterns:
        if re.search(pat, query_lower):
            return False
            
    # 2. Check positive keywords
    for kw in positive_kws:
        if match_keyword(kw, query_lower):
            return True
            
    return None


def has_dangerous_weapon_or_means_check(query_lower: str) -> bool:
    english_weapons = [
        "weapon", "weapons", "gun", "pistol", "firearm", "knife", "knives", "blade", "blades", 
        "sword", "swords", "dagger", "axe", "machete", "iron rod", "metal bar", "hammer", "bat", 
        "fire", "poison", "acid", "corrosive", "chemical", "noxious", "explosive", "boiling water", 
        "hot water", "pour hot", "stick", "sticks", "pole", "poles"
    ]
    for kw in english_weapons:
        if match_keyword(kw, query_lower):
            return True

    sinhala_weapons = [
        "ආයුධ", "පිහි", "කඩු", "තුවක්කු", "යකඩ පොල්ල", "ගිනි", "ගින්දර", "ගින්න", "ඇසිඩ්", "රසායනික", 
        "පුපුරණ", "පොල්ල", "කෝටු"
    ]
    for kw in sinhala_weapons:
        if match_keyword(kw, query_lower):
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
    
    # Check explicit victim age key-value
    match_kv = re.search(r"(?:victim\s*age|age)\s*[:=]?\s*(\d{1,2})", query_lower)
    if match_kv:
        return int(match_kv.group(1))

    # Check English age formats (e.g. 14-year-old, 14 year-old, 14 years old, aged 14)
    match_en = re.search(
        r"(\d{1,2})\s*[-–—\s]*\s*(?:years?|yrs?)[-–—\s]+old"
        r"|(\d{1,2})\s*[-–—\s]*\s*(?:years?|yrs?)\s+of\s+age"
        r"|aged\s*(\d{1,2})",
        query_lower
    )
    if match_en:
        for g in match_en.groups():
            if g:
                return int(g)

    # Check Sinhala age formats (e.g., වයස අවුරුදු 14, අවුරුදු 14, 14 හැවිරිදි, 14ක දරුවෙක්)
    match_si = re.search(r"(?:වයස\s*අවුරුදු|වයස|අවුරුදු)\s*(\d{1,2})", query_lower)
    if match_si:
        return int(match_si.group(1))

    match_si_haviridi = re.search(r"(\d{1,2})\s*හැවිරිදි", query_lower)
    if match_si_haviridi:
        return int(match_si_haviridi.group(1))
        
    match_si_suffix = re.search(r"(\d{1,2})\s*ක\s*(?:දරුවෙක්|ළමයෙක්|ගැහැණු|පිරිමි|පුද්ගලයෙක්|කාන්තාවක්|මිනිසෙක්|කෙනෙක්|දැරියක්)", query_lower)
    if match_si_suffix:
        return int(match_si_suffix.group(1))

    return None


def extract_all_structured_facts(query: str, language: str = None) -> dict:
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
        "traffic_restricted_articles": None,
        "extortion": None
    }

    # 1. victim_age
    facts["victim_age"] = extract_victim_age(query)

    # 2. victim_sex
    female_kws = ["girl", "daughter", "female", "sister", "she", "her", "niece", "granddaughter", "ගැහැණු", "දුව", "කාන්තාව", "ඇය", "ඇයට", "ඇයව", "ඇයගේ"]
    male_kws = ["boy", "son", "male", "brother", "he", "him", "nephew", "grandson", "පිරිමි", "පුතා", "මිනිසා", "ඔහු", "ඔහුට", "ඔහුව", "ඔහුගේ"]
    
    is_female = any(match_keyword(kw, query_lower) for kw in female_kws)
    is_male = any(match_keyword(kw, query_lower) for kw in male_kws)
    if is_female and not is_male:
        facts["victim_sex"] = "female"
    elif is_male and not is_female:
        facts["victim_sex"] = "male"

    # 3. offender_relationship
    rel_map = {
        "parent": ["father", "mother", "parent", "parents", "dad", "mom", "stepfather", "stepmother", "පියා", "මව", "තාත්තා", "අම්මා", "දෙමාපිය", "දෙමව්පිය", "බාප්පා", "කුඩම්මා", "ලොකු තාත්තා", "ලොකු අම්මා", "පුංචි අම්මා"],
        "guardian": ["guardian", "custodian", "භාරකාර", "භාරකරු", "භාරකාරත්වය", "භාරකාරත්වය දරන", "භාරව සිටින"],
        "caregiver": ["caregiver", "nanny", "babysitter", "maid", "රැකබලා ගන්නා", "රැකවරණය භාර", "භාරව සිටින තැනැත්තා", "භාරකරු", "භාරකාරිණිය"],
        "teacher": ["teacher", "instructor", "warden", "principal", "ගුරු", "ගුරුවරයා", "ගුරුවරිය", "විදුහල්පති"],
        "employer": ["employer", "boss", "master", "ස්වාමියා", "හාම්පුතා"],
        "stranger": ["stranger", "unknown person", "unknown", "outsider", "intruder", "unfamiliar", "අමුත්තෙක්", "අඳුනන්නේ නැති", "නුහුරු", "අමුතු"],
        "relative": ["uncle", "aunt", "cousin", "relative", "relatives", "family", "ඥාති", "ඥාතියා", "ඥාතීන්", "මාමා", "නැන්දා"]
    }
    
    for rel, kws in rel_map.items():
        if any(match_keyword(kw, query_lower) for kw in kws):
            facts["offender_relationship"] = rel
            break

    # 4. custody_or_care
    custody_pos = ["custody", "charge", "care", "responsible for", "taking care of", "in the care of", "caregiver", "guardian", "at home", "රැකවරණය", "භාරව", "භාරයේ", "රැකබලා", "භාරකාරත්වය", "භාරකරු", "නිවසේදී"]
    custody_neg = ["no custody", "not in care", "stranger", "non-caregiver"]
    
    if facts["offender_relationship"] in ["parent", "guardian", "caregiver"]:
        facts["custody_or_care"] = True
    elif facts["offender_relationship"] == "stranger":
        facts["custody_or_care"] = False
    else:
        custody_val = check_fact_presence_and_negation(query_lower, custody_pos, [rf"\b{re.escape(n)}\b" for n in custody_neg])
        facts["custody_or_care"] = custody_val

    # Clean threats/warned-not-to-tell out of query for physical violence checks to avoid false positives
    query_clean_phys = query_lower
    threat_phrases_phys = [
        "threat of harm", "threat of hurt", "threat of injury", "threat of violence",
        "threatened with harm", "threatened with hurt", "threatened with violence",
        "threatened harm", "threatened to harm", "threatened to hurt", "threatened to beat", "threatened to hit",
        "threaten to harm", "threaten to hurt", "threaten to beat", "threaten to hit",
        "හානියක් කරන බවට තර්ජනය", "හානියක් කරන බවට", "පහර දෙන බවට තර්ජනය", "පහර දෙන බවට",
        "මරන බවට තර්ජනය", "මරන බවට", "රිදවන බවට තර්ජනය", "මරණ තර්ජන", "බියවැද්දීම"
    ]
    for phrase in threat_phrases_phys:
        query_clean_phys = query_clean_phys.replace(phrase, "")

    # Define binary fact patterns mapping
    binary_patterns = {
        "physical_assault": (
            ["hit", "beat", "beaten", "struck", "assault", "assaulted", "slap", "slapped", "punch", "punched", "kick", "kicked", "beating", "physically harmed", "physical harm", "physical abuse", "physically abused", "lash", "lashed", "whip", "whipped", "strike", "cane", "caned", "caning", "slaps", "blow", "blows", "violence", "physical violence", "stab", "stabbed", "stabbing", "cut", "cutting", "slash", "slashing", "injure", "injures", "injured", "torture", "tortured", "පහර", "ගැහුවා", "ගහනවා", "ගහලා", "බැට", "හිංසනය", "තැළුවා", "තලනවා", "ඇන්නා", "ඇනීම", "කැපුවා", "වේවැල්", "කෝටු", "ගුටි", "තැලීම", "කෲර", "වධහිංසා", "හිංසා", "අතින් පහර", "කම්මුල් පහර", "පහර දීම"],
            [r"\b(?:no|without|did not|never|free of)\s+(?:physical\s+)?(?:assault|beating|hitting|striking|abuse|violence|harm)\b", r"\b(?:assault|beating|hitting|striking|abuse)\s+(?:was not|did not|never occurred|is absent)\b", r"පහරදීමක්\s+(?:සිදු\s+වී\s+)?නැත", r"පහර\s+දී\s+නැත", r"පහර\s+දුන්නේ\s+නැත", r"ගැසීමක්\s+සිදු\s+වී\s+නැත", r"හිංසනයක්\s+සිදු\s+වී\s+නැත", r"හිංසා\s+කර\s+නැත", r"කෲර\s+ලෙස\s+සලකා\s+නැත", r"ගුටි\s+දී\s+නැත"]
        ),
        "physical_injury": (
            ["injury", "injured", "wound", "wounded", "bleeding", "fracture", "bruise", "bruises", "pain", "visible injuries", "harm", "swelling", "swollen", "cuts", "laceration", "scar", "scars", "broken bone", "broken tooth", "broken", "fractured", "fracturing", "painful", "bruised", "bleeding", "තුවාල", "ලේ ගැලීම", "තැල්ම", "නිල් තැල්ම", "වේදනාව", "වේදනාවන්", "වේදනා", "ශාරීරික හානි", "ඇඟ රිදෙනවා", "කැක්කුම", "ඉදිමීම", "ඉදිමීම්", "ඉදිමුම්", "තැලීම", "තැලීම්", "රිදවීම", "රිදවීම්", "රිදෙව්වා", "කැඩුණු"],
            [r"\b(?:no|without|free of|did not cause|no bodily)\s+(?:visible\s+)?(?:injury|injuries|bruise|bruises|wound|wounds|swelling|bleeding|pain|harm)\b", r"\b(?:injury|injuries|bruise|bruises|wound|wounds|swelling|bleeding|pain|harm)\s+(?:were not|did not|was not|not found|not present|absent)\b", r"(?:තුවාල|තුවාලයක්|වේදනාවක්|හානියක්)\s+(?:[^\n]{0,30})\s+(?:නැත|නොමැත|නොවීය)", r"තුවාල\s+(?:සිදු\s+වී\s+)?නැත", r"තුවාලයක්\s+නැත", r"තුවාල\s+නැත", r"ශාරීරික\s+හානියක්\s+නැත", r"වේදනාවක්\s+නැත"]
        ),
        "sexual_contact": (
            ["touch", "touched", "touching", "private parts", "groped", "indecent touch", "inappropriate touch", "sexual touch", "fondle", "fondled", "fondling", "breast", "breasts", "genital", "genitals", "butt", "buttocks", "thigh", "thighs", "අනුචිත ලෙස ස්පර්ශ", "අනවශ්‍ය ලෙස ස්පර්ශ", "අයුතු ලෙස ස්පර්ශ", "අතපත ගෑම", "ස්පර්ශ කළ", "ස්පර්ශ කිරීම", "ස්පර්ශ", "වැරදි විදියට", "වැරදි ලෙස ස්පර්ශ", "රහස් ප්‍රදේශ", "ලිංගික ප්‍රදේශ", "රහස් කොටස්"],
            [r"\b(?:no|without|did not|never)\s+(?:sexual\s+)?(?:touch|touching|private parts|groping|fondling)\b", r"\b(?:touch|touching|groping|fondling)\s+(?:was not|did not|never occurred|is absent)\b", r"ස්පර්ශ\s+කර\s+නැත", r"ස්පර්ශයක්\s+(?:සිදු\s+)?නොවීය", r"ස්පර්ශයක්\s+නැත", r"ස්පර්ශ\s+කිරීමක්\s+(?:සිදු\s+වී\s+)?නැත"]
        ),
        "sexual_act": (
            ["sexual act", "sexual acts", "sexual abuse", "sexually abuse", "sexually abused", "sexual conduct", "sexual nature", "sexual violation", "grave sexual abuse", "sexual crime", "sexual assault", "sexually assaulted", "sexually assault", "sexually assaulting", "ලිංගික ක්‍රියා", "ලිංගික වධදීම", "ලිංගික අපයෝජනය", "අතවරයකට"],
            [r"\b(?:no|without)\s+(?:sexual\s+)?(?:acts?|abuse|assault|violation)\b", r"ලිංගික\s+ක්‍රියාවක්\s+සිදු\s+කර\s+නැත", r"ලිංගික\s+අපයෝජනයක්\s+(?:සිදු\s+වී\s+)?නැත"]
        ),
        "penetration": (
            ["rape", "raped", "penetration", "forced intercourse", "forced sex", "penetrated", "statutory rape", "intercourse", "carnal intercourse", "sexual intercourse", "දූෂණය", "ලිංගික සංසර්ගය", "බලහත්කාරයෙන් ලිංගික", "සංසර්ගය"],
            [r"\b(?:no|without|did not|never)\s+(?:explicit\s+|sexual\s+)?(?:intercourse|penetration|penetrating|rape)\b", r"\b(?:intercourse|penetration|penetrating|rape)\s+(?:was not|did not|never occurred|is absent)\b", r"ලිංගික\s+සංසර්ගයක්\s+(?:සිදු\s+)?නොවීය", r"සංසර්ගයක්\s+(?:සිදු\s+)?නොවීය", r"ඇතුල්\s+කිරීමක්\s+(?:සිදු\s+)?නොවීය", r"දූෂණය\s+කර\s+නැත", r"ඇතුල්\s+කර\s+නැත", r"ලිංගික\s+සංසර්ගයක්\s+හෝ\s+ඇතුල්\s+කිරීමක්\s+(?:ප්‍රකාශ\s+කර\s+නැත|සිදු\s+නොවීය)"]
        ),
        "sexual_harassment": (
            ["sexual harassment", "modesty", "unwelcome sexual", "sexual comments", "catcall", "outrage modesty", "harass", "ලිංගික හිරිහැර", "ලිංගික අතවර", "ලැජ්ජාවට පත්", "අශෝභන කතා"],
            [r"\b(?:no|without)\s+(?:sexual\s+)?harassment\b", r"ලිංගික\s+හිරිහැරයක්\s+නැත"]
        ),
        "sexual_image_material": (
            ["photo", "photos", "video", "videos", "picture", "pictures", "csam", "obscene", "nude", "media", "recording", "publish photo", "upload photo", "camera", "ඡායාරූප", "වීඩියෝ", "පින්තූර", "අසභ්‍ය", "නිරුවත්", "කැමරා"],
            [r"\b(?:no|without|did not take|never shared)\s+(?:\w+\s+){0,3}(?:photo|video|picture|material|nude|image)s?\b", r"ඡායාරූප\s+හෝ\s+වීඩියෝ\s+(?:ගෙන\s+)?නැත", r"ඡායාරූප\s+නැත", r"වීඩියෝ\s+නැත"]
        ),
        "online_contact": (
            ["computer", "internet", "online", "website", "platform", "server", "isp", "service provider", "app", "digital", "social media", "telegram", "whatsapp", "facebook", "messenger", "viber", "imo", "පරිගණක", "අන්තර්ජාලය", "වෙබ්", "ඔන්ලයින්", "සේවා සපයන්නා"],
            [r"\b(?:not\s+online|not\s+via\s+internet|offline|no\s+internet|did\s+not\s+use\s+computer)\b", r"\b(?:not|never|did not)\s+(?:\w+\s+){0,3}(?:online|internet|computer|social media|whatsapp|facebook|viber|telegram|share)\b", r"අන්තර්ජාලය\s+භාවිතා\s+කර\s+නැත", r"ඔන්ලයින්\s+නොවේ"]
        ),
        "kidnapping": (
            ["kidnap", "kidnapped", "kidnapping", "lawful guardianship", "snatch", "පැහැරගැනීම", "පැහැරගෙන", "භාරකාරත්වයෙන් පැහැර"],
            [r"\b(?:no\s+kidnapping|was\s+not\s+kidnapped)\b", r"පැහැරගෙන\s+නැත", r"පැහැරගැනීමක්\s+සිදු\s+වී\s+නැත"]
        ),
        "taking_from_guardian": (
            ["took the child away", "took away", "enticed away", "enticing from", "take from lawful", "භාරකාරත්වයෙන් බැහැර", "රැගෙන ගියා", "රවටා රැගෙන", "රැගෙන යාම", "රැගෙන යාමට"],
            []
        ),
        "abduction": (
            ["abduct", "abducted", "forcefully taken", "compelled by force", "compell", "abduction", "බලහත්කාරයෙන් රැගෙන", "පැහැරගෙන", "බලහත්කාරයෙන් රැගෙන ගියා", "බලහත්කාරයෙන්"],
            [r"\b(?:no\s+abduction|was\s+not\s+abducted)\b", r"බලහත්කාරයෙන්\s+රැගෙන\s+ගොස්\s+නැත"]
        ),
        "trafficking": (
            ["traffic", "trafficking", "sold", "buying", "selling", "transported for exploitation", "human trafficking", "child trafficking", "recruitment", "recruit", "recruited", "transit", "harbor", "harbored", "receipt of a child", "receipt of child", "receives", "received", "ජාවාරම", "විකිණීම", "ළමා ජාවාරම", "ගනුදෙනු", "මිනිස් ජාවාරම"],
            [r"\b(?:no\s+trafficking|not\s+trafficked|never\s+sold)\b", r"ළමා\s+ජාවාරමක්\s+නොවේ", r"විකිණීමක්\s+සිදු\s+වී\s+නැත"]
        ),
        "commercial_exploitation": (
            ["procurer", "prostitution", "brothel", "commercial sex", "solicit", "soliciting", "grooming", "pimp", "prostitute", "sex trade", "sex work", "තැරැව්කාර", "ප්‍රසම්පාදක", "ලිංගික සූරාකෑම", "ගණිකා", "පොළඹවා ගැනීම"],
            []
        ),
        "begging": (
            ["beg", "begging", "alms", "beggar", "solicit alms", "සිඟමන්", "හිඟා", "සිඟමන් යැදීම", "හිඟමන්"],
            [r"\b(?:not\s+begging|no\s+begging)\b", r"සිඟමන්\s+යැදීමක්\s+නොවේ"]
        ),
        "neglect": (
            ["neglect", "without care", "without protection", "no care", "no protection", "not cared for", "not looked after", "neglected", "failure to provide", "නොසලකා", "නොසලකා හැරීම", "ආරක්ෂාව නැහැ", "රැකවරණයක් නැති", "නොසලකා හරියි", "රැකවරණය නොමැති"],
            [r"\b(?:no\s+neglect|proper\s+care|looked\s+after\s+well)\b", r"නොසලකා\s+හැරීමක්\s+නැත", r"රැකවරණය\s+ලබා\s+දී\s+ඇත"]
        ),
        "food_deprivation": (
            ["without food", "no food", "starved", "starving", "no food and water", "food deprivation", "nothing to eat", "hunger", "කෑම නැති", "නිරාහාරව", "කෑම බීම නොදී", "කෑම නොදී", "කෑම ඉල්ලා", "කෑම ඉල්ලන", "කෑම ඉල්ලයි", "බඩගින්නේ", "බඩගිනි"],
            [r"\b(?:proper\s+food|not\s+starved)\b", r"නිරාහාරව\s+තබා\s+නැත"]
        ),
        "medical_neglect": (
            ["medical neglect", "no medical", "without medical care", "refused medical treatment", "failed to seek medical", "වෛද්‍ය ප්‍රතිකාර නොදී", "බෙහෙත් නොදී", "ප්‍රතිකාර නොකර"],
            []
        ),
        "lack_of_supervision": (
            ["left alone", "unattended", "unsupervised", "no supervision", "බැලීමට කෙනෙකු නොමැතිව", "තනිවම දමා", "තනිව දාලා", "කිසිවෙකු නොමැතිව"],
            []
        ),
        "abandonment": (
            ["abandon", "abandoned", "deserted", "left alone in public", "intent to desert", "desertion", "අත්හැර", "අත්හැර දමා", "දමා ගොස්", "අතහැර දමා"],
            [r"\b(?:not\s+abandoned|did\s+not\s+abandon)\b", r"අත්හැර\s+දමා\s+නැත", r"දමා\s+ගොස්\s+නැත"]
        ),
        "intent_to_wholly_abandon": (
            ["wholly abandon", "intent to wholly", "intent to desert", "permanently left", "සම්පූර්ණයෙන්ම අත්හැර"],
            []
        ),
        "health_suffering": (
            ["causing suffering", "cause suffering", "injury to health", "suffering to health", "හානියක් සිදුකිරීම", "පීඩාවක් ඇතිකිරීම", "හඬා", "හඬමින්", "වැලපෙමින්", "crying", "weeping"],
            []
        ),
        "threats": (
            ["threatened", "warned not to tell", "frightened into silence", "threat", "threats", "threaten", "threatens", "threat of harm", "threat of injury", "threat of violence", "death threat", "threatened into silence", "coerced into silence", "forced into silence", "intimidated", "silenced", "coerce", "coercion", "තර්ජනය", "තර්ජන", "තර්ජනයක්", "තර්ජනය කළ", "තර්ජනය කර", "කිසිවෙකුට නොකියන ලෙස තර්ජනය", "නොකියන ලෙස තර්ජනය", "මරා දමන බවට තර්ජනය", "පහර දෙන බවට තර්ජනය", "හානියක් කරන බවට තර්ජනය", "බියගන්වා නිහඬ කිරීම", "බියගන්වා", "බියවැද්දීම", "බියගැන්වීම", "බිය ගැන්වූ"],
            [r"\b(?:no\s+threats|did\s+not\s+threaten)\b", r"තර්ජනයක්\s+කර\s+නැත"]
        ),
        "confinement": (
            ["confinement", "confined", "locked inside", "locked in a room", "imprisoned", "wahuwa", "hira", "හිරකර", "කොටු කර", "වසා තිබූ"],
            [r"\b(?:not\s+confined|never\s+locked\s+inside)\b", r"හිරකර\s+තබා\s+නැත"]
        ),
        "sexual_touching": (
            ["touch", "touched", "touching", "private parts", "groped", "indecent touch", "inappropriate touch", "fondle", "fondled", "fondling", "genitals", "butt", "buttocks", "thigh", "thighs", "අනුචිත ලෙස ස්පර්ශ", "අනවශ්‍ය ලෙස ස්පර්ශ", "අතපත ගෑම", "ස්පර්ශ කළ", "ස්පර්ශ කිරීම", "ස්පර්ශ", "වැරදි විදියට", "වැරදි ලෙස ස්පර්ශ", "රහස් කොටස්", "අයුතු ලෙස ස්පර්ශ"],
            [r"\b(?:no|without|did not|never)\s+(?:sexual\s+)?(?:touch|touching|private parts|fondling|groping)\b", r"ස්පර්ශ\s+කර\s+නැත", r"ස්පර්ශයක්\s+සිදු\s+නොවීය", r"ස්පර්ශ\s+කිරීමක්\s+නැත"]
        ),
        "repeated_conduct": (
            ["repeated", "repeatedly", "multiple times", "often", "frequently", "ongoing", "continuous", "over and over", "several times", "නැවත නැවතත්", "නිරන්තරයෙන්", "පිට පිට", "බොහෝ වාරයක්", "නිරතුරුවම", "දිගින් දිගටම", "නිතර නිතර", "නිතර"],
            []
        ),
        "adult_offender": (
            ["adult", "grown-up", "man", "woman", "uncle", "aunt", "parent", "caregiver", "guardian", "known adult", "adult offender", "වැඩිහිටි", "වැඩිහිටියෙකු", "වැඩිහිටියා"],
            []
        ),
        "threat_to_keep_silent": (
            ["keep silent", "keep quiet", "dont tell", "don't tell", "not to tell", "warned not to tell", "threatened to keep silent", "silence", "tells anyone", "if you tell", "secret", "keep it secret", "නිශ්ශබ්දව", "නොකියන ලෙස", "නොකියන ලෙසට", "කියන්න එපා", "කිසිවෙකුට නොකියන"],
            []
        ),
        "threat_of_harm": (
            ["threat of harm", "threatened with harm", "threaten to harm", "threaten to beat", "kill", "hurt", "threat of violence", "threaten to injure", "harm if", "හානියක් කරන", "හානි කරන", "පහර දෙන බවට තර්ජනය", "මරණ තර්ජන", "මරන බවට", "මරනවා", "තර්ජනය"],
            []
        ),
        "psychological_distress": (
            ["distress", "fear", "fearful", "scared", "afraid", "terrified", "traumatized", "depression", "anxiety", "psychological", "emotional pain", "mental suffering", "trauma", "frightened", "බිය", "බය", "බියට", "බියෙන්", "බියට පත්", "බියට පත්ව", "බියෙන් සිටී", "බියපත්ව", "මානසික පීඩාව", "මානසික කෲරත්වය", "මානසික පීඩා"],
            []
        ),
        "unnatural_intercourse": (
            ["unnatural carnal", "buggery", "against the order of nature", "anal sex", "oral sex", "sodomy", "ස්වභාවධර්මයට පටහැනි", "ගුද සංසර්ගය", "මුඛ සංසර්ගය", "අස්වාභාවික ලිංගික"],
            []
        ),
        "sodomy": (
            ["sodomy", "sodomized", "buggery", "anal sex", "ගුද සංසර්ගය"],
            []
        ),
        "gross_indecency": (
            ["gross indecency", "grossly indecent", "gross indecency act", "බරපතල අශෝභන ක්‍රියා", "අශෝභන ක්‍රියා"],
            []
        ),
        "employ_child_as_procurer": (
            ["employing children to act as procurers", "hiring children to act as procurers", "employ a child as a procurer", "hire a child as a procurer", "act as a procurer", "procurer", "තැරැව්කරුවන් ලෙස ළමයින් යොදා ගැනීම", "තැරැව්කරුවන්"],
            []
        ),
        "traffic_restricted_articles": (
            ["traffic in restricted articles", "trafficking restricted articles", "restricted articles", "sell drugs", "sell liquor", "තහනම් භාණ්ඩ ජාවාරම", "තහනම් ද්‍රව්‍ය", "මත්ද්‍රව්‍ය"],
            []
        ),
        "extortion": (
            ["extort", "extortion", "extorting", "extorted", "demand money", "demanding money", "demand property", "demanding property", "valuable security", "force payment", "ransom", "demanded cash", "compel illegal act", "constrain to illegal act", "කප්පම්", "කප්පම් ගැනීම", "දේපළ ලබාගැනීම", "මුදල් බලහත්කාරයෙන් ලබාගැනීම", "මුදල් ඉල්ලා", "දේපල ලබාගැනීම", "විරෝධී ක්‍රියාවකට බලකිරීම"],
            [r"\b(?:no\s+extortion|did\s+not\s+demand\s+money|no\s+money\s+demanded)\b", r"කප්පම්\s+ගැනීමක්\s+නැත"]
        )
    }

    # Evaluate all binary patterns
    for fact_key, (pos_kws, neg_pats) in binary_patterns.items():
        if fact_key == "physical_assault":
            # For physical assault, evaluate using clean phys query to avoid threat false positives
            facts[fact_key] = check_fact_presence_and_negation(query_clean_phys, pos_kws, neg_pats)
        elif fact_key == "physical_injury":
            # Same for physical injury
            facts[fact_key] = check_fact_presence_and_negation(query_clean_phys, pos_kws, neg_pats)
        else:
            facts[fact_key] = check_fact_presence_and_negation(query_lower, pos_kws, neg_pats)

    # 5. Special checks & Inferences
    # weapon_or_dangerous_means override
    weapon_negations = [
        "no weapon", "no weapon was used", "did not use a weapon", "did not use a knife", 
        "without weapon", "ආයුධ භාවිතා කළේ නැත", "ආයුධයක් තිබුණේ නැත", "පොල්ලක් භාවිතා කලේ නැත"
    ]
    if any(neg in query_lower for neg in weapon_negations):
        facts["weapon_or_dangerous_means"] = False
    elif has_dangerous_weapon_or_means_check(query_lower):
        facts["weapon_or_dangerous_means"] = True

    # injury_severity logic
    grievous_indicators = [
        "emasculation", "impotent", "castration", "නපුංසක", "වන්ධ්‍යා",
        "blind", "sight", "deaf", "hearing", "අන්ධ", "පෙනීම", "බිහිරි", "ඇසීම",
        "limb", "joint", "amputation", "amputate", "severed", "අතපය", "අත් පා", "සන්ධි",
        "disfigure", "disfigurement", "scar", "facial", "විකෘති",
        "fracture", "fractured", "fracturing", "dislocate", "dislocated", "dislocating", "bone broken", "broken bone", 
        "broken tooth", "teeth broken", "tooth knocked", "knocked out tooth", "breaking", "broken",
        "බිඳී", "බිඳීම", "බිඳීම්", "පැනීම", "කැඩී", "හැලී",
        "endanger life", "endangers life", "life-threatening", "critical condition", "icu", "coma", 
        "20 days", "twenty days", "දින 20", "දවස් 20", "මරණාසන්න"
    ]
    is_grievous = any(match_keyword(kw, query_clean_phys) for kw in grievous_indicators)
    if is_grievous:
        facts["injury_severity"] = "grievous"
    elif facts["physical_injury"] is True:
        facts["injury_severity"] = "simple"

    # begging special check
    # Context check: if begging father/mother/parents/caregiver, it is NOT alms begging
    begging_context_en = r"beg(?:ging)?\s+(?:his\s+|her\s+|their\s+)?(?:father|mother|parent|parents|guardian|caregiver|teacher)"
    begging_context_si = r"(?:පියාගෙන්|මවගෙන්|දෙමාපියන්ගෙන්|ඥාතීන්ගෙන්)\s+(?:කෑම\s+)?ඉල්ලා"
    if re.search(begging_context_en, query_lower) or re.search(begging_context_si, query_lower):
        facts["begging"] = False

    # intent_to_wholly_abandon logic
    if facts["abandonment"] is True and facts["custody_or_care"] is True:
        if facts["intent_to_wholly_abandon"] is None:
            facts["intent_to_wholly_abandon"] = True

    # intercourse override based on penetration
    if facts["penetration"] is True and facts["intercourse"] is None:
        facts["intercourse"] = True
    elif facts["penetration"] is False:
        facts["intercourse"] = False

    # employ_child_as_procurer commercial exploitation link
    if facts["commercial_exploitation"] is True and "procur" in query_lower:
        facts["employ_child_as_procurer"] = True

    # secondary inferences
    if facts["food_deprivation"] is True or facts["medical_neglect"] is True or facts["lack_of_supervision"] is True:
        if facts["neglect"] is None or facts["neglect"] is False:
            facts["neglect"] = True

    if facts["physical_assault"] is True:
        if facts["health_suffering"] is None:
            facts["health_suffering"] = True

    # adult_offender logic
    if facts["offender_relationship"] in ["parent", "guardian", "caregiver", "teacher", "employer", "relative"]:
        facts["adult_offender"] = True

    # Robust list-negation overrides (Requirement 9 & 14)
    if any(neg in query_lower for neg in ["no kidnapping", "no abduction", "no taking", "without kidnapping", "without abduction", "පැහැරගෙන නැත", "පැහැරගැනීමක් සිදු වී නැත", "රැගෙන ගොස් නැත"]):
        facts["kidnapping"] = False
        facts["abduction"] = False
        facts["taking_from_guardian"] = False

    if any(neg in query_lower for neg in ["no intercourse", "no penetration", "without intercourse", "without penetration", "ලිංගික සංසර්ගයක් සිදු නොවීය", "ඇතුල් කිරීමක් සිදු නොවීය", "ලිංගික සංසර්ගයක් හෝ ඇතුල් කිරීමක් ප්‍රකාශ කර නොමැත", "ලිංගික සංසර්ගයක් හෝ ඇතුල් කිරීමක් ප්‍රකාශ කර නැත", "ලිංගික සංසර්ගයක් සිදු වී නැත"]):
        facts["penetration"] = False
        facts["intercourse"] = False

    if any(neg in query_lower for neg in ["no touch", "no touching", "without touch", "without touching", "ස්පර්ශ කර නැත", "ස්පර්ශයක් නැත"]):
        facts["sexual_contact"] = False
        facts["sexual_touching"] = False

    if any(neg in query_lower for neg in ["no photo", "no video", "no media", "without photo", "without video", "ඡායාරූප නැත", "වීඩියෝ නැත", "ඡායාරූප හෝ වීඩියෝ ගෙන නැත"]):
        facts["sexual_image_material"] = False

    if any(neg in query_lower for neg in ["no online", "not online", "offline", "not via internet", "අන්තර්ජාලය භාවිතා කර නැත", "ඔන්ලයින් නොවේ"]):
        facts["online_contact"] = False

    if any(neg in query_lower for neg in ["no begging", "not begging", "සිඟමන් යැදීමක් නොවේ"]):
        facts["begging"] = False

    if any(neg in query_lower for neg in ["no injury", "no injuries", "no wound", "no wounds", "no pain", "without injury", "without pain", "තුවාලයක් නැත", "තුවාල නැත", "වේදනාවක් නැත", "තුවාලයක් හෝ ශාරීරික වේදනාවක් සිදු නොවීය", "තුවාලයක් හෝ ශාරීරික වේදනාවක් සිදු නැත"]):
        facts["physical_injury"] = False

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
    if facts_dict["kidnapping"] is True or facts_dict["abduction"] is True or facts_dict["taking_from_guardian"] is True:
        extracted_canonical.append("kidnapping")
    if facts_dict["trafficking"] is True:
        extracted_canonical.append("trafficking")
    if facts_dict["commercial_exploitation"] is True:
        extracted_canonical.append("commercial_exploitation")
    if facts_dict["begging"] is True:
        extracted_canonical.append("begging")
    if facts_dict["neglect"] is True or facts_dict["food_deprivation"] is True or facts_dict["medical_neglect"] is True or facts_dict["lack_of_supervision"] is True or facts_dict["health_suffering"] is True or facts_dict["psychological_distress"] is True:
        extracted_canonical.append("neglect")
    if facts_dict["abandonment"] is True:
        extracted_canonical.append("abandonment")
    if facts_dict["threats"] is True or facts_dict["threat_of_harm"] is True or facts_dict["threat_to_keep_silent"] is True:
        extracted_canonical.append("threats")
    if facts_dict.get("extortion") is True:
        extracted_canonical.append("extortion")

    return sorted(list(set(extracted_canonical)))
