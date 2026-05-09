import os
import io
import wave
import json
import numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from .listener import phone_audio_listener
from .predictor import predictor
from core.supabase import db

PARENT_PROFILE_PATH = Path("parent_profile.wav")
CONFIG_PATH = Path("audio_guardian_config.json")

router = APIRouter()

def get_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"parent_name": "Parent"}

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)

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
    config = get_config()
    parent_name = config.get("parent_name", "Parent")
    
    # Calculate raw volume (dB) to demonstrate solution to Intensity Bias
    try:
        with wave.open(io.BytesIO(contents), 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
            amplitude_db = 20 * np.log10(rms) if rms > 0 else 0.0
    except Exception:
        amplitude_db = 0.0

    # Run the Deep Learning Model
    class_id, probability = predictor.predict_from_wav_bytes(contents)
    
    # Class_ID 1 = Threat (Scream, Aggression), 0 = Safe (Normal noises)
    status_msg = "Safe"
    mitigation_msg = None
    
    # Always check if Parent is speaking if there's enough volume
    is_parent = False
    if amplitude_db > 45.0:
        is_parent = predictor.verify_parent(contents, PARENT_PROFILE_PATH)
        
    if class_id == 1:
        if is_parent:
            status_msg = f"Safe ({parent_name} is speaking - Threat Override)"
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
    else:
        # If class is 0 (Safe), but we recognized the parent
        if is_parent:
            status_msg = f"Safe ({parent_name} is speaking)"
            
    if class_id == 0 and amplitude_db > 75.0 and not is_parent:
        mitigation_msg = f"Anti-Fatigue Activated: Loud noise ({amplitude_db:.1f}dB) detected, but AI confirmed it as SAFE. Alert Suppressed!"
        
    return {
        "filename": file.filename,
        "class_id": class_id,
        "status": status_msg,
        "probability": f"{probability:.2%}",
        "amplitude_db": round(amplitude_db, 2),
        "mitigation_message": mitigation_msg
    }

@router.post("/register-parent")
async def register_parent_voice(
    file: UploadFile = File(...),
    parent_name: str = Form("Parent")
):
    """
    Endpoint to register a baseline parent voice profile.
    Saves the WAV file and updates the parent name dynamically.
    """
    contents = await file.read()
    with open(PARENT_PROFILE_PATH, "wb") as f:
        f.write(contents)
        
    config = get_config()
    config["parent_name"] = parent_name
    save_config(config)
    
    return {
        "message": "Parent voice profile saved successfully.",
        "parent_name": parent_name,
        "filename": file.filename
    }

@router.post("/clear-alerts")
async def clear_alerts():
    """
    Endpoint to clean dirty test data from the database.
    """
    try:
        # Delete all records where id is not null (which deletes all rows)
        db.table('audio_threat_alerts').delete().neq('sensor_type', 'dummy').execute()
        return {"message": "Test data cleared successfully."}
    except Exception as e:
        return {"error": f"Failed to clear data: {str(e)}"}
