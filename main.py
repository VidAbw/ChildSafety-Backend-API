# main.py
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from audio_guardian import router as audio_router
from nanny_cam_guardian import router as nanny_router
from app.api.v1.router import api_v1_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Child Safety Guardian API",
    description="Backend API for Child Safety Guardian monitoring, telemetry ingestion, and alert management.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend origin, e.g. http://localhost:8081
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect the existing domain routers
app.include_router(nanny_router.router, prefix="/api/iot", tags=["Nanny Cam Guardian (MM-ODG)"])
app.include_router(audio_router, prefix="/api/audio", tags=["Audio Guardian"])

# Connect the new Safety Alert Pipeline API (v1)
app.include_router(api_v1_router, prefix="/api/v1")


@app.get("/")
def health_check():
    return {
        "status": "online",
        "message": "Child Safety Guardian API is running",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
