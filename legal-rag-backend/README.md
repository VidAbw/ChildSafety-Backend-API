# Legal RAG Backend

A FastAPI backend for a Retrieval-Augmented Generation system for child protection legal guidance in Sri Lanka.

## Setup

1. Create virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables in .env

4. Run the server:
   ```
   uvicorn app.main:app --reload
   ```

5. Access docs at http://127.0.0.1:8000/docs

## Features

- Abuse type classification
- Legal section retrieval using RAG
- Decision roadmap generation
- Privacy-preserving (no raw descriptions stored)