from app.services.rag_service import build_faiss_index, load_legal_sections
import os

if __name__ == "__main__":
    try:
        print("Loading legal sections...")
        sections = load_legal_sections()
        print(f"Loaded {len(sections)} sections.")
        print("Building FAISS index...")
        build_faiss_index(sections)
        print("FAISS index rebuilt successfully.")
    except Exception as e:
        print(f"Error: {e}")
