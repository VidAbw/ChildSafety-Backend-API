import json
import os
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.schemas.legal_schema import LegalSection
from app.schemas.rag_schema import RelevantLaw

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'legal_sections_penal.json')
INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'legal_index.faiss')
IDS_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'ids.json')
TEXTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store', 'texts.json')
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

_model = None
_index = None
_sections = None
_ids = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_legal_sections() -> List[LegalSection]:
    global _sections
    if _sections is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Legal sections dataset not found at {DATA_PATH}")
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _sections = [LegalSection(**item) for item in data]
    return _sections


def save_legal_sections(sections: List[LegalSection]) -> None:
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump([section.dict() for section in sections], f, ensure_ascii=False, indent=2)
    global _sections
    _sections = sections


def load_faiss_index():
    global _index, _ids
    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError('FAISS index not found. Run app/vector_store/build_index.py first.')
        _index = faiss.read_index(INDEX_PATH)
        with open(IDS_PATH, 'r', encoding='utf-8') as f:
            _ids = json.load(f)
    return _index, _ids


def build_faiss_index(sections: List[LegalSection] = None) -> None:
    if sections is None:
        sections = load_legal_sections()

    if not sections:
        raise ValueError('No legal sections available to build FAISS index.')

    model = get_model()
    texts = [
        f"{section.law_name} {section.section_number} {section.legal_text_summary} {section.simple_explanation}"
        for section in sections
    ]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, 0)
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(IDS_PATH, 'w', encoding='utf-8') as f:
        json.dump([section.id for section in sections], f, ensure_ascii=False, indent=2)
    with open(TEXTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    global _index, _ids
    _index = index
    _ids = [section.id for section in sections]


def import_legal_sections(sections: List[LegalSection], rebuild_index: bool = True) -> None:
    save_legal_sections(sections)
    if rebuild_index:
        build_faiss_index(sections)


def retrieve_relevant_laws(query: str, abuse_category: str, top_k: int = 3) -> List[RelevantLaw]:
    sections = load_legal_sections()

    try:
        index, ids = load_faiss_index()
        model = get_model()
        query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype('float32')
        faiss.normalize_L2(query_embedding)

        distances, indices = index.search(query_embedding, top_k)
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(ids):
                continue
            section_id = ids[idx]
            section = next((s for s in sections if s.id == section_id), None)
            if section:
                results.append(RelevantLaw(
                    section=section.section_number,
                    title=f"{section.law_name} {section.section_number}",
                    simple_explanation=section.simple_explanation,
                    reporting_guidance=section.reporting_guidance,
                    relevance_score=round(float(distances[0][rank]), 3)
                ))
        if results:
            return results
    except Exception as e:
        print(f"FAISS search failed: {e}")

    fallback = [
        s for s in sections
        if s.abuse_category and abuse_category.lower() in s.abuse_category.lower()
    ]
    if not fallback:
        fallback = sections[:top_k]

    return [
        RelevantLaw(
            section=s.section_number,
            title=f"{s.law_name} {s.section_number}",
            simple_explanation=s.simple_explanation,
            reporting_guidance=s.reporting_guidance,
            relevance_score=None,
        )
        for s in fallback[:top_k]
    ]