import json
import os
import urllib.parse
import urllib.request
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "penal.json"

def translate_text(text: str, sl: str = "en", tl: str = "si") -> str:
    if not text or not text.strip():
        return ""
    
    text_clean = text.strip()
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={sl}&tl={tl}&dt=t&q={urllib.parse.quote(text_clean)}"
    
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    translations = [part[0] for part in data[0] if part and part[0]]
                    result = "".join(translations)
                    return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for text: {text[:30]}... Error: {e}")
            time.sleep(1)
            
    return ""

def main():
    if not INPUT_PATH.exists():
        print(f"File not found: {INPUT_PATH}")
        return
        
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        sections = json.load(f)
        
    print(f"Loaded {len(sections)} sections from {INPUT_PATH}")
    
    modified = False
    translated_count = 0
    
    for i, section in enumerate(sections):
        # Fields to translate
        title_en = section.get("title") or ""
        # Try both simple_explanation_en and simple_explanation
        exp_en = section.get("simple_explanation_en") or section.get("simple_explanation") or ""
        rep_en = section.get("reporting_guidance") or ""
        
        title_si = section.get("title_si") or ""
        exp_si = section.get("simple_explanation_si") or ""
        rep_si = section.get("reporting_guidance_si") or ""
        
        need_save = False
        
        if not title_si:
            print(f"[{i+1}/{len(sections)}] Translating title: {title_en}")
            translated_title = translate_text(title_en)
            if translated_title:
                section["title_si"] = translated_title
                need_save = True
                translated_count += 1
                time.sleep(0.3)
                
        if not exp_si:
            print(f"[{i+1}/{len(sections)}] Translating explanation: {exp_en}")
            translated_exp = translate_text(exp_en)
            if translated_exp:
                section["simple_explanation_si"] = translated_exp
                need_save = True
                translated_count += 1
                time.sleep(0.3)
                
        if not rep_si:
            print(f"[{i+1}/{len(sections)}] Translating reporting guidance: {rep_en}")
            translated_rep = translate_text(rep_en)
            if translated_rep:
                section["reporting_guidance_si"] = translated_rep
                need_save = True
                translated_count += 1
                time.sleep(0.3)
                
        if need_save:
            modified = True
            # Save intermediate results in case script is interrupted
            with open(INPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
                
    print(f"Translation complete. Translated {translated_count} fields.")

if __name__ == "__main__":
    main()
