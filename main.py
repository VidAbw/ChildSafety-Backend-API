# main.py
import logging

from fastapi import FastAPI
from audio_guardian import router as audio_router
from audio_guardian.listener import phone_audio_listener
from nanny_cam_guardian import router as nanny_router
# from chat_counselor import router as chat_router       # Member 3 — AI Counselor
# from game import router as game_router                 # Member 4 — S-ALS Game

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Child Safety Guardian API")

# Connect the routers
app.include_router(nanny_router.router, prefix="/api/iot", tags=["Nanny Cam Guardian (MM-ODG)"])
app.include_router(audio_router, prefix="/api/audio", tags=["Audio Guardian"])
# app.include_router(chat_router.router, prefix="/api/chat", tags=["AI Counselor"])
# app.include_router(game_router.router, prefix="/api/game", tags=["S-ALS Game"])


@app.on_event("startup")
async def startup_event() -> None:
    await phone_audio_listener.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await phone_audio_listener.stop()

@app.get("/")
def health_check():
    return {
        "status": "online",
        "message": "Backend is running",
        "audio_listener": phone_audio_listener.status(),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)