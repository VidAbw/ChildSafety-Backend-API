from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.health import router as health_router
from app.routers.rag import router as rag_router
from app.routers.legal import router as legal_router

app = FastAPI(
    title="Legal RAG Backend",
    description="API for child protection legal guidance using RAG",
    version="1.0.0"
)

# CORS middleware for Expo app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Expo app URL
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