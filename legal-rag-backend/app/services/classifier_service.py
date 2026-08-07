from typing import Tuple, List

def classify_abuse_categories(text: str) -> Tuple[str, List[str]]:
    text = text.lower()

    # Priority mapping matching verified child-abuse categories
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

    matched_categories = []
    for category, keywords in categories.items():
        if any(word in text for word in keywords):
            matched_categories.append(category)

    if not matched_categories:
        return "general_child_protection", []

    primary = matched_categories[0]
    secondary = [c for c in matched_categories[1:] if c != primary]
    return primary, secondary


def classify_abuse(text: str) -> str:
    primary, _ = classify_abuse_categories(text)
    return primary
