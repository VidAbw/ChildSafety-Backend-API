from fastapi import APIRouter

from .listener import phone_audio_listener

router = APIRouter()


@router.get("/status")
def get_audio_listener_status() -> dict:
    return phone_audio_listener.status()


@router.post("/start")
async def start_audio_listener() -> dict:
    await phone_audio_listener.start()
    return {
        "message": "Audio listener start requested.",
        "status": phone_audio_listener.status(),
    }


@router.post("/stop")
async def stop_audio_listener() -> dict:
    await phone_audio_listener.stop()
    return {
        "message": "Audio listener stopped.",
        "status": phone_audio_listener.status(),
    }
