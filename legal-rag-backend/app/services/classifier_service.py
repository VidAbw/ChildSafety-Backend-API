def classify_abuse(text: str) -> str:
    text = text.lower()

    neglect_keywords = [
        "neglect",
        "abandon",
        "abandoned",
        "left alone",
        "without food",
        "without care",
        "without protection",
        "no food",
        "no care",
        "no protection",
        "not cared for",
        "not looked after"
    ]

    physical_keywords = [
        "beat",
        "beaten",
        "hit",
        "harm",
        "harmed",
        "injury",
        "injured",
        "physically harmed",
        "physical abuse",
        "hurt"
    ]

    sexual_keywords = [
    "sexual",
    "rape",
    "indecent",
    "exploit",
    "exploitation",
    "sexual abuse",
    "indecent photos",
    "obscene",
    "child photos",
    "obscene photos",
    "sexual images",
    "sexual content",
    "grooming",
    "lure",
    "luring",
    "solicit",
    "soliciting"
    ]

    trafficking_keywords = [
        "traffic",
        "trafficking",
        "moved for exploitation",
        "transported",
        "controlled for exploitation",
        "sold",
        "forced labour",
        "slavery"
    ]

    digital_keywords = [
        "online",
        "internet",
        "computer",
        "photos",
        "videos",
        "platform",
        "digital abuse"
    ]

    if any(word in text for word in neglect_keywords):
        return "neglect"

    if any(word in text for word in physical_keywords):
        return "physical abuse"

    if any(word in text for word in sexual_keywords):
        return "sexual abuse"

    if any(word in text for word in trafficking_keywords):
        return "trafficking"

    if any(word in text for word in digital_keywords):
        return "digital abuse"

    return "general abuse"