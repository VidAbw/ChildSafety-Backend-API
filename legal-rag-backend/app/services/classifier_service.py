from typing import List

# Simple keyword-based classifier
ABUSE_CATEGORIES = {
    "physical abuse": ["hit", "beat", "slap", "punch", "kick", "harm", "injure", "bruise", "wound"],
    "sexual abuse": ["touch", "molest", "rape", "abuse", "assault", "inappropriate", "naked", "sex"],
    "neglect": ["neglect", "starve", "hungry", "dirty", "unfed", "uncared", "abandon"],
    "digital abuse": ["online", "internet", "cyber", "harass", "bully", "threat", "stalk"],
    "emotional abuse": ["yell", "scare", "threaten", "insult", "humiliate", "ignore", "reject"]
}

def classify_abuse(description: str) -> str:
    description_lower = description.lower()
    for category, keywords in ABUSE_CATEGORIES.items():
        if any(keyword in description_lower for keyword in keywords):
            return category
    return "general abuse"  # Default