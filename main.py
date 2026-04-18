# main.py
from fastapi import FastAPI
from nanny_cam_guardian import router as nanny_router
# from audio_guardian import router as audio_router      # Member 2 — Audio Guardian
# from chat_counselor import router as chat_router       # Member 3 — AI Counselor
# from game import router as game_router                 # Member 4 — S-ALS Game

app = FastAPI(title="Child Safety Central Logic Hub")

# Connect the routers
app.include_router(nanny_router.router, prefix="/api/hub", tags=["Central Logic Hub"])
# app.include_router(audio_router.router, prefix="/api/audio", tags=["Audio Guardian"])
# app.include_router(chat_router.router, prefix="/api/chat", tags=["AI Counselor"])
# app.include_router(game_router.router, prefix="/api/game", tags=["S-ALS Game"])

@app.get("/")
def health_check():
    return {"status": "online", "message": "Backend is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)