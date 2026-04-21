from fastapi import APIRouter

from .listener import phone_audio_listener

router = APIRouter()


@router.get("/status")
def get_audio_listener_status() -> dict:
    return phone_audio_listener.status()
