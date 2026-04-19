from langdetect import detect, LangDetectError

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        if lang == 'si':
            return 'si'  # Sinhala
        elif lang == 'en':
            return 'en'  # English
        else:
            return 'en'  # Default to English
    except LangDetectError:
        return 'en'