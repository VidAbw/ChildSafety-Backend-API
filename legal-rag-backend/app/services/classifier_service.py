from typing import Tuple, List
from app.services.fact_extraction_service import extract_all_structured_facts

def classify_abuse_categories(text: str) -> Tuple[str, List[str]]:
    text_lower = text.lower()
    
    # 1. Extract structured facts
    facts = extract_all_structured_facts(text)
    
    matched_categories = []
    
    # 2. Fact-based classification
    # 2.1 Sexual Abuse
    if (facts.get("penetration") is True or facts.get("intercourse") is True or 
        facts.get("sexual_contact") is True or facts.get("sexual_touching") is True or 
        facts.get("sexual_act") is True or facts.get("unnatural_intercourse") is True or 
        facts.get("sodomy") is True or facts.get("gross_indecency") is True):
        matched_categories.append("sexual_abuse")
        
    # 2.2 Physical Abuse
    if facts.get("physical_assault") is True or facts.get("physical_injury") is True:
        matched_categories.append("physical_abuse")
        
    # 2.3 Cruelty
    if facts.get("repeated_conduct") is True or (facts.get("custody_or_care") is True and facts.get("physical_assault") is True):
        matched_categories.append("cruelty")
        
    # 2.4 Neglect
    if (facts.get("neglect") is True or facts.get("food_deprivation") is True or 
        facts.get("medical_neglect") is True or facts.get("lack_of_supervision") is True or 
        facts.get("abandonment") is True or facts.get("begging") is True):
        matched_categories.append("neglect")
        
    # 2.5 Sexual Exploitation
    if facts.get("commercial_exploitation") is True or facts.get("employ_child_as_procurer") is True:
        matched_categories.append("sexual_exploitation")
        
    # 2.6 Trafficking
    if facts.get("trafficking") is True:
        matched_categories.append("trafficking")
        
    # 2.7 Kidnapping Abduction
    if facts.get("kidnapping") is True or facts.get("abduction") is True or facts.get("taking_from_guardian") is True:
        matched_categories.append("kidnapping_abduction")
        
    # 2.8 Online or Material Abuse
    if facts.get("sexual_image_material") is True or facts.get("online_contact") is True:
        matched_categories.append("online_or_material_abuse")
        
    # 2.9 Sexual Harassment
    if facts.get("sexual_harassment") is True:
        matched_categories.append("sexual_harassment")
        
    # 2.10 Emotional Abuse
    if (facts.get("threats") is True or facts.get("threat_of_harm") is True or 
        facts.get("threat_to_keep_silent") is True or facts.get("psychological_distress") is True or 
        facts.get("confinement") is True):
        matched_categories.append("emotional_abuse")

    # 3. Fallback to keyword-based matching if no fact-based categories matched
    if not matched_categories:
        categories = {
            "online_or_material_abuse": [
                "photo", "photos", "video", "videos", "picture", "pictures", "camera", "record", 
                "recording", "publish", "share", "image", "images", "media", "telegram", "whatsapp", 
                "social media", "exhibition", "distribution", "csam", "obscene", "computer", "internet",
                "online", "website", "web", "platform", "server", "isp", "service provider", "app",
                "ඡායාරූප", "වීඩියෝ", "පින්තූර", "කැමරා", "මුද්‍රණය", "ප්‍රදර්ශනය", "ප්‍රකාශන", "පරිගණක", "අන්තර්ජාලය"
            ],
            "sexual_harassment": [
                "sexual harassment", "modesty", "unwelcome sexual", "sexual comments", "catcall",
                "ලිංගික හිරිහැර", "ලිංගික අතවර"
            ],
            "sexual_exploitation": [
                "procurer", "procure", "prostitution", "brothel", "sex work", "exploitation", "solicit",
                "soliciting", "grooming", "lure", "luring", "pornographic", "commercial sex",
                "තැරැව්කාර", "ප්‍රසම්පාදක", "ලිංගික සූරාකෑම"
            ],
            "kidnapping_abduction": [
                "kidnap", "kidnapped", "abduct", "abducted", "entice", "taken away", "lawful guardianship",
                "පැහැරගැනීම", "භාරකාරත්වයෙන්"
            ],
            "cruelty": [
                "cruelty", "corporal punishment", "ill-treat", "ill-treatment", "willfully assault",
                "කෲර", "කෲර ලෙස"
            ],
            "physical_abuse": [
                "beat", "beaten", "hit", "harm", "harmed", "injury", "injured", 
                "physically harmed", "physical abuse", "hurt", "assault",
                "wound", "bleeding", "fracture", "weapon", "slap", "bruise",
                "වධ හිංසා", "වධහිංසා", "පහර දුන්නා", "ගැහුවා", "පහරදීම", "තුවාල", 
                "ලේ ගැලීම", "ශාරීරික හානි", "බැට දුන්නා", "පහර"
            ],
            "neglect": [
                "neglect", "abandon", "abandoned", "left alone", "without food", 
                "without care", "without protection", "no food", "no care", 
                "no protection", "not cared for", "not looked after", "starved", "starving", "beg", "begging",
                "නොසලකා", "නොසලකා හරියි", "නොසලකා හැරීම", "කෑම නැහැ", 
                "ආරක්ෂාව නැහැ", "තනිව දාලා", "අත්හැර", "රැකවරණයක් නැති",
                "කන්න බොන්න", "රැකවරණය", "දමා ගොස්", "අත්හැර දමා", "සිඟමන්", "හිඟා"
            ],
            "trafficking": [
                "traffic", "trafficking", "moved for exploitation", "transported", 
                "controlled for exploitation", "sold", "forced labour", "forced labor", "slavery", 
                "debt bondage", "adoption", "child soldier", "armed conflict",
                "ගනුදෙනු", "වහල්", "ජාවාරම", "විකිණීම", "විදේශ", "බලහත්කාර", "දරුකමට"
            ],
            "sexual_abuse": [
                "sexual", "rape", "indecent", "sexual abuse", "incest",
                "touching", "touched", "private parts", "inappropriate touch", "penetration", "statutory rape", "touch", "touched inappropriately",
                "ලිංගික අපයෝජනය", "අසභ්ය ලෙස ස්පර්ශ කිරීම", "අසභ්‍ය ලෙස ස්පර්ශ කිරීම", 
                "දූෂණය", "අතපත ගෑම", "ඥාති සංවාසය", "ලිංගික", "අසභ්‍ය",
                "අනුචිත ලෙස ස්පර්ශ", "අනවශ්‍ය ලෙස ස්පර්ශ", "ස්පර්ශ", "ස්පර්ශ කළ", "ලිංගික අතවර"
            ],
            "emotional_abuse": [
                "emotional", "mental", "trauma", "shouting", "insulting", "bullying",
                "harassment", "scare", "scared", "fear", "threat", "threatening",
                "depressed", "depression", "crying", "anxiety", "counselor", "counseling",
                "therapy", "therapist", "mental health", "trauma help", "suicidal", "sadness", "distress", "unsafe",
                "මානසික සහනයක්", "අඬනවා", "කනස්සල්ල", "තෙරපි", "මානසික උපදේශනය",
                "මානසික", "බියවැද්දීම", "තර්ජනය", "කෑගැසීම", "අපහාස",
                "මානසිකව", "බිය", "මානසික පීඩාව", "ආරක්ෂාවක් නැති"
            ]
        }
        for category, keywords in categories.items():
            if any(word in text_lower for word in keywords):
                matched_categories.append(category)

    if not matched_categories:
        return "general_child_protection", []

    primary = matched_categories[0]
    return primary, []


def classify_abuse(text: str) -> str:
    primary, _ = classify_abuse_categories(text)
    return primary