def classify_abuse(text: str) -> str:
    text = text.lower()

    # Mapping keys to broader categories that match legal_sections.json
    categories = {
        "sexual_abuse": [
            "sexual", "rape", "indecent", "exploit", "exploitation", "sexual abuse", 
            "indecent photos", "obscene", "child photos", "obscene photos", 
            "sexual images", "sexual content", "grooming", "lure", "luring", 
            "solicit", "soliciting", "incest", "prostitution", "brothel", "sex work",
            "ලිංගික අපයෝජනය", "ලිංගික අතවර", "අසභ්ය ලෙස ස්පර්ශ කිරීම", "අසභ්‍ය ලෙස ස්පර්ශ කිරීම", 
            "දූෂණය", "ලිංගික හිරිහැර"
        ],
        "physical_abuse": [
            "beat", "beaten", "hit", "harm", "harmed", "injury", "injured", 
            "physically harmed", "physical abuse", "hurt", "assault", "cruelty",
            "වධ හිංසා", "වධහිංසා", "පහර දුන්නා", "ගැහුවා", "පහරදීම", "තුවාල", 
            "ලේ ගැලීම", "ශාරීරික හානි", "බැට දුන්නා", "කෲර ලෙස සැලකීම"
        ],
        "neglect": [
            "neglect", "abandon", "abandoned", "left alone", "without food", 
            "without care", "without protection", "no food", "no care", 
            "no protection", "not cared for", "not looked after",
            "නොසලකා", "නොසලකා හරියි", "නොසලකා හැරීම", "කෑම නැහැ", 
            "ආරක්ෂාව නැහැ", "තනිව දාලා", "අත්හැර", "රැකවරණයක් නැති",
            "කන්න බොන්න", "රැකවරණය", "දමා ගොස්"
        ],
        "trafficking": [
            "traffic", "trafficking", "moved for exploitation", "transported", 
            "controlled for exploitation", "sold", "forced labour", "slavery", 
            "kidnap", "kidnapped", "abduction", "abducted", "debt bondage",
            "ගනුදෙනු", "පැහැරගැනීම", "වහල්", "ජාවාරම", "විකිණීම", "විදේශ"
        ],
        "emotional_abuse": [
            "emotional", "mental", "trauma", "shouting", "insulting", "bullying",
            "harassment", "scare", "scared", "fear", "threat", "threatening",
            "depressed", "depression", "crying", "anxiety", "counselor", "counseling",
            "therapy", "therapist", "mental health", "trauma help", "suicidal", "sadness",
            "මානසික සහනයක්", "අඬනවා", "කනස්සල්ල", "තෙරපි", "මානසික උපදේශනය",
            "මානසික", "බියවැද්දීම", "තර්ජනය", "හිරිහැර", "කෑගැසීම", "අපහාස",
            "මානසිකව", "බිය"
        ]
    }

    for category, keywords in categories.items():
        if any(word in text for word in keywords):
            return category

    return "general_child_protection"