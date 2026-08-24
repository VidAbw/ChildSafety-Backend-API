# pyrefly: ignore [missing-import]
import sys
import os

# Ensure legal-rag-backend directory is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_rag_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _rag_root not in sys.path:
    sys.path.insert(0, _rag_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.routers.health import router as health_router
    from app.routers.rag import router as rag_router
    from app.routers.legal import router as legal_router
except (ImportError, ModuleNotFoundError):
    from routers.health import router as health_router
    from routers.rag import router as rag_router
    from routers.legal import router as legal_router


app = FastAPI(
    title="Legal RAG Backend",
    description="API for child protection legal guidance using RAG",
    version="1.0.0"
)

# CORS middleware for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(rag_router)
app.include_router(legal_router)

@app.get("/")
async def root():
    return {"message": "Welcome to Legal RAG Backend"}