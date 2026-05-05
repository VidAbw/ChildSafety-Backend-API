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

def generate_roadmap(abuse_category: str) -> List[str]:
    return ROADMAP_TEMPLATES.get(abuse_category, ROADMAP_TEMPLATES["general abuse"])