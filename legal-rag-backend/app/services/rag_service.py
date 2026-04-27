import json
import os
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.schemas.legal_schema import LegalSection
from app.schemas.rag_schema import RelevantLaw

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'legal_sections.json')
INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'legal_index.faiss')
TEXTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'texts.json')
MODEL_NAME = 'all-MiniLM-L6-v2'


def load_legal_sections() -> List[LegalSection]:
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [LegalSection(**item) for item in data]


def save_legal_sections(sections: List[LegalSection]) -> None:
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump([s.dict() for s in sections], f, ensure_ascii=False, indent=2)


def load_faiss_index() -> Tuple[faiss.IndexFlatL2, List[str]]:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(TEXTS_PATH):
        raise FileNotFoundError('FAISS index or texts file not found')

    index = faiss.read_index(INDEX_PATH)
    with open(TEXTS_PATH, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    return index, texts


def build_faiss_index() -> None:
    sections = load_legal_sections()
    texts = [f"{s.law_name} {s.section_number}: {s.legal_text_summary} {s.simple_explanation}" for s in sections]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts)
    embeddings = np.asarray(embeddings).astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(TEXTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)


def retrieve_relevant_laws(query: str, abuse_category: str, top_k: int = 3) -> List[RelevantLaw]:
    sections = load_legal_sections()
    relevant_sections: List[LegalSection] = []

    try:
        index, _ = load_faiss_index()
        model = SentenceTransformer(MODEL_NAME)
        query_embedding = model.encode([query])
        query_embedding = np.asarray(query_embedding).astype('float32')

        distances, indices = index.search(query_embedding, top_k)
        relevant_sections = [sections[i] for i in indices[0] if 0 <= i < len(sections)]
    except Exception:
        relevant_sections = []

    if not relevant_sections:
        relevant_sections = [s for s in sections if s.abuse_category == abuse_category]

    if not relevant_sections:
        relevant_sections = sections[:top_k]

    return [
        RelevantLaw(
            section=f"{s.law_name} {s.section_number}",
            title=s.law_name,
            simple_explanation=s.simple_explanation
        ) for s in relevant_sections[:top_k]
    ]


def import_legal_sections(sections: List[LegalSection], rebuild_index: bool = True) -> None:
    save_legal_sections(sections)
    if rebuild_index:
        build_faiss_index()
