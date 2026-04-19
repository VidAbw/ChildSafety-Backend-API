import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.services.rag_service import load_legal_sections

def build_faiss_index():
    sections = load_legal_sections()
    texts = [f"{s.law_name} {s.section_number}: {s.legal_text_summary} {s.simple_explanation}" for s in sections]

    model = SentenceTransformer('all-MiniLM-L6-v2')  # Small model for prototype
    embeddings = model.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))

    # Save index
    faiss.write_index(index, os.path.join(os.path.dirname(__file__), 'legal_index.faiss'))

    # Save texts for retrieval
    with open(os.path.join(os.path.dirname(__file__), 'texts.json'), 'w') as f:
        json.dump(texts, f)

    print("FAISS index built and saved.")

if __name__ == "__main__":
    build_faiss_index()