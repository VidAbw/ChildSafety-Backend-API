# main.py
from fastapi import FastAPI
from routers import iot, chat, game # Import your team's files

app = FastAPI(title="Child Safety Guardian API")

# Connect the routers
app.include_router(iot.router, prefix="/api/iot", tags=["IoT Guardian"])
# app.include_router(chat.router, prefix="/api/chat", tags=["AI Counselor"]) 
# app.include_router(game.router, prefix="/api/game", tags=["S-ALS Game"])

@app.get("/")
def health_check():
    return {"status": "online", "message": "Backend is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)