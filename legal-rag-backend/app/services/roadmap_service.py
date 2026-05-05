from typing import List

ROADMAP_TEMPLATES = {
    "physical abuse": [
        "Ensure the child's immediate safety by removing them from the harmful situation if possible.",
        "Do not confront the suspected abuser directly to avoid escalation.",
        "Contact the National Child Protection Authority (NCPA) hotline at 1929.",
        "Report to the Sri Lanka Police emergency service at 119.",
        "Gather evidence safely, such as photos of injuries, without putting yourself at risk.",
        "Seek medical attention for the child if needed."
    ],
    "sexual abuse": [
        "Ensure the child's safety and avoid further contact with the suspected abuser.",
        "Do not bathe or change the child's clothes if medical examination is needed.",
        "Contact NCPA hotline 1929 immediately.",
        "Report to Police 119 for urgent assistance.",
        "Seek medical help from a hospital equipped to handle such cases.",
        "Preserve any evidence and avoid discussing details publicly."
    ],
    "neglect": [
        "Assess the child's immediate needs for food, shelter, and medical care.",
        "Contact NCPA 1929 to report the neglect.",
        "If the child is in immediate danger, call Police 119.",
        "Document signs of neglect with photos if safe.",
        "Seek support from local child welfare services."
    ],
    "digital abuse": [
        "Block the abuser on all digital platforms.",
        "Save evidence such as messages, screenshots, without responding.",
        "Report to Police 119 or Cyber Crime Unit.",
        "Contact NCPA 1929 for guidance.",
        "Change passwords and enable two-factor authentication.",
        "Monitor the child's online activity closely."
    ],
    "emotional abuse": [
        "Provide emotional support to the child.",
        "Document instances of abuse with dates and details.",
        "Contact NCPA 1929 for counseling and reporting.",
        "If threats are involved, report to Police 119.",
        "Seek professional help for the child's mental health."
    ],
    "general abuse": [
        "Ensure the child's safety first.",
        "Contact NCPA hotline 1929.",
        "Report to Police emergency 119 if urgent.",
        "Keep records of incidents.",
        "Consult legal experts for further guidance."
    ]
}

ROADMAP_TEMPLATES_SI = {
    "physical abuse": [
        "දරුවාගේ වහාම ආරක්ෂාව සනාථ කර බලශක්තිමත් අවස්ථාවකින් හානිදායක පරිසරයෙන් ඉවත් කරන්න.",
        "කියවුවාවරුන්ට සෘජුවම වෙහෙස නොකිරීමට හෝ තත්ත්වය තවත් අසීරු කර ගැනීමට වළක්වන්න.",
        "1929 අංකයෙන් ජාතික ළමා ආරක්ෂක අධිකාරිය අමතන්න.",
        "ඉක්මන් සහාය සඳහා 119 අංකයෙන් ශ්‍රී ලංකා පොලිස් හදිසි සේවාවට වාර්තා කරන්න.",
        "කෙළින්ම අනතුරට ලක් නොවෙන ලෙස ඡායා චායාරූප වැනි සාක්ෂි ආරක්ෂා කරන්න.",
        "දරුවාට අවශ්‍ය නම් වෛද්‍ය සැත්කම් ලබා දෙන්න."
    ],
    "sexual abuse": [
        "දරුවාගේ ආරක්ෂාව සහාය කරන්න සහ සැකගත් ප්‍රහාරකයා සමඟ තවත් සම්බන්ධතා වළක්වන්න.",
        "වෛද්‍ය පරීක්ෂණ අවශ්‍ය නම් දරුවාගේ ඇඳුම් හෝ සෝදන ආදාන නොකරන්න.",
        "පෙරමුණෙන්ම 1929 අංකයෙන් NCPA ට ඇමතුමක් කරන්න.",
        "හාදිසි සහාය සඳහා 119 අංකයෙන් පොලිසියට වාර්තා කරන්න.",
        "මෙවැනි හේතු සඳහා සුදුසු රෝහලකින් වෛද්‍ය සහාය ලබා ගන්න.",
        "සියලු සාක්ෂි ඔරොත්තු දමා පල 공개 නොකරන්න."
    ],
    "neglect": [
        "ආහාර, නවාතැන සහ වෛද්‍ය සත්කාරය සඳහා දරුවාගේ වහාම අවශ්‍යතා ඇස්තමේන්තුව කරන්න.",
        "අවම වශයෙන් 1929 අංකයෙන් NCPA ට මෙය වාර්තා කරන්න.",
        "දරුවා වහාම අවදානමට පත්ව තිබේ නම් 119 අංකයෙන් පොලිසියට අමතන්න.",
        "ආරක්ෂිතව හැකි නම් අවදානම් ලක්ෂණ ඡායාරූප ලබා ගන්න.",
        "ප්‍රදේශීය ළමා සුභසාධන සේවාවලින් සහාය ලබන්න."
    ],
    "digital abuse": [
        "සියලු ඩිජිටල් වේදිකාවලදී ප්‍රහාරකයා අවහිර කරන්න.",
        "පණිවිඩ, ඡායා තිර, වැනි සාක්ෂි සුරකින්න, ප්‍රතිචාර නොදෙන්න.",
        "119 හෝ අන්තර්ජාල අපරාධ ඒකකය වෙත වාර්තා කරන්න.",
        "මාර්ගෝපදේශ සඳහා 1929 අංකයෙන් NCPA ට අමතන්න.",
        "මුරපද වෙනස් කර දෙයි-තත්කාලික සත්‍යාපන ආරක්ෂාව සක්‍රිය කරන්න.",
        "දරුවාගේ අන්තර්ජාල කටයුතු සැලකිල්ලෙන් නිරීක්ෂණය කරන්න."
    ],
    "emotional abuse": [
        "දරුවාට මානසික සහය ලබා දෙන්න.",
        "දින හා විස්තර සමඟ ප්‍රහාර සිදුවීම් ලේඛනය කරන්න.",
        "ආරක්ෂණය සහ වාර්තාව සඳහා 1929 අංකයෙන් NCPA ට අමතන්න.",
        "ආපදා තර්ජන ඇත්නම් 119 අංකයෙන් වාර්තා කරන්න.",
        "දරුවාගේ මානසික සෞඛ්‍ය සඳහා වෘත්තීය උදව් ලබා ගන්න."
    ],
    "general abuse": [
        "ප්‍රථමයෙන් දරුවාගේ ආරක්ෂාව අවශ්‍යයි.",
        "1929 අංකයෙන් NCPA ට අමතන්න.",
        "ඉක්මනින් අවශ්‍ය නම් 119 අංකයෙන් පොලිසු අමතන්න.",
        "සිද්ධීන් සසා බැලීමේ ලේඛන තබා ගන්න.",
        "වැඩි දුර උපදෙස් සඳහා නීතිඥයින් සම්බන්ධ වන්න."
    ]
}


def generate_roadmap(abuse_category: str, language: str = "en") -> List[str]:
    templates = ROADMAP_TEMPLATES_SI if language.lower() == "si" else ROADMAP_TEMPLATES
    return templates.get(abuse_category, templates["general abuse"])
