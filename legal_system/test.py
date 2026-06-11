import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load the Legal Knowledge Base 
with open('legal_system/legal_data.json', 'r') as f:
    legal_docs = json.load(f)

# 2. Initialize DistilBERT Model [cite: 206, 267]
# This model converts text into semantic embeddings
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# 3. Create Embeddings for all legal descriptions [cite: 196, 198]
descriptions = [doc['description'] for doc in legal_docs]
embeddings = model.encode(descriptions)

# 4. Initialize FAISS for Similarity Search [cite: 167, 199, 207]
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

def get_legal_guidance(user_query):
    # Convert user description to an embedding [cite: 196]
    query_embedding = model.encode([user_query])
    
    # Search FAISS for the top match [cite: 198, 207]
    D, I = index.search(np.array(query_embedding).astype('float32'), k=1)
    
    # Retrieve the document [cite: 318]
    best_match_idx = I[0][0]
    return legal_docs[best_match_idx]

# --- TEST THE PROTOTYPE ---
sample_query = "A neighbor is hitting their child repeatedly and the child is crying."
result = get_legal_guidance(sample_query)

print(f"Detected Category: {result['category']}")
print(f"Relevant Law: Section {result['section']} - {result['title']}")
print(f"Advice: {result['punishment']}")