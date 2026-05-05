import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from .listener import phone_audio_listener
from .predictor import predictor

PARENT_PROFILE_PATH = Path("parent_profile.wav")

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

@router.post("/upload-chunk")
async def upload_audio_chunk(
    file: UploadFile = File(...),
    device_info: str = Form("unknown")
):
    """
    Endpoint for a microphone streamer to upload 3-second audio chunks.
    Runs the 1D-CNN + LSTM model and triggers an alert if it's a Threat.
    """
    contents = await file.read()
    
    # Run the Deep Learning Model
    class_id, probability = predictor.predict_from_wav_bytes(contents)
    
    # Class_ID 1 = Threat (Scream, Aggression), 0 = Safe (Normal noises)
    status_msg = "Safe"
    if class_id == 1:
        # Before alerting, verify if it's the parent's voice
        is_parent = predictor.verify_parent(contents, PARENT_PROFILE_PATH)
        
        if is_parent:
            status_msg = "Safe (Parent Voice Verified)"
            class_id = 0
        else:
            # Moderate if < 85%, High if >= 85%
            threat_level = "high" if probability >= 0.85 else "moderate"
            status_msg = f"Threat Detected ({threat_level.capitalize()})"
            # Trigger an alert in Supabase
            phone_audio_listener._trigger_supabase_alert(
                intensity_score=probability * 100.0, 
                threat_level=threat_level,
                device_info=device_info
            )
        
    return {
        "filename": file.filename,
        "class_id": class_id,
        "status": status_msg,
        "probability": f"{probability:.2%}"
    }

@router.post("/register-parent")
async def register_parent_voice(file: UploadFile = File(...)):
    """
    Endpoint to register a baseline parent voice profile.
    Send a 5-10 second WAV file of the parent speaking.
    """
    contents = await file.read()
    with open(PARENT_PROFILE_PATH, "wb") as f:
        f.write(contents)
    
    return {
        "message": "Parent voice profile saved successfully.",
        "filename": file.filename
    }
