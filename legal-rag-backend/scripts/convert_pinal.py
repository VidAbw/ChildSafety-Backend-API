import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "pinal.json"
OUTPUT_PATH = ROOT / "app" / "data" / "legal_sections_penal.json"
DEFAULT_DATA_PATH = ROOT / "app" / "data" / "legal_sections.json"


def normalize_section(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(raw.get("id", "")),
        "law_name": raw.get("law_name", ""),
        "section_number": str(raw.get("section_number", "")),
        "abuse_category": raw.get("category", raw.get("sub_category", "")),
        "legal_text_summary": raw.get("legal_text_summary_en", raw.get("legal_text_summary", "")),
        "simple_explanation": raw.get("simple_explanation_en", raw.get("simple_explanation", "")),
        "keywords": raw.get("keywords", []),
        "reporting_guidance": raw.get("reporting_guidance", ""),
        "source": raw.get("source_document", raw.get("source", "")),
    }


def load_input(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_output(data: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert pinal.json to RAG legal section schema.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="Path to the source pinal.json file.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Path to the converted output JSON file.")
    parser.add_argument("--overwrite", action="store_true", help="Also overwrite app/data/legal_sections.json with the converted data.")
    args = parser.parse_args()

    raw_sections = load_input(args.input)
    converted = [normalize_section(item) for item in raw_sections]
    save_output(converted, args.output)
    print(f"Converted {len(converted)} sections to {args.output}")

    if args.overwrite:
        save_output(converted, DEFAULT_DATA_PATH)
        print(f"Overwrote {DEFAULT_DATA_PATH}")


if __name__ == "__main__":
    main()
