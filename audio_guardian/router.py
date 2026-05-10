import os
import io
import json
import logging
import numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from .listener import phone_audio_listener
from .predictor import predictor
from core.supabase import db

logger = logging.getLogger(__name__)

PARENT_PROFILE_PATH = Path("parent_profile.wav")
CONFIG_PATH = Path("audio_guardian_config.json")

router = APIRouter()

# In-memory store of the latest ESP32 prediction (so the frontend can poll it)
_last_result: dict = {}

# When set, the next audio chunk from the ESP32 is captured as a voice registration
_register_next_for: dict = {}  # keys: person_name, role

# ──────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────
def get_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"parent_name": "Parent"}

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)

# ──────────────────────────────────────────────────────────────
# Load all active MFCC profiles from Supabase
# ──────────────────────────────────────────────────────────────
def get_registered_mfcc_profiles() -> list:
    """
    Fetches all active voice profiles from Supabase registered_voice_profiles table.
    Returns a list of MFCC 2D arrays ready for DTW comparison.
    """
    try:
        result = db.table("registered_voice_profiles") \
            .select("dtw_feature_matrix, person_name, role") \
            .eq("is_active", True) \
            .execute()

        rows = result.data or []
        logger.info(f"Loaded {len(rows)} active voice profile(s) from Supabase.")
        matrices = []
        for row in rows:
            matrix = row.get("dtw_feature_matrix")
            if matrix is not None:
                matrices.append(matrix)
                logger.info(f"  - Profile: {row.get('person_name')} ({row.get('role')})")
        return matrices
    except Exception as e:
        logger.warning(f"Could not load profiles from Supabase: {e}")
        return []

# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────
@router.get("/status")
def get_audio_listener_status() -> dict:
    config = get_config()
    try:
        profiles = db.table("registered_voice_profiles").select("id").eq("is_active", True).execute()
        profile_count = len(profiles.data or [])
    except Exception:
        profile_count = 0

    return {
        "backend": "online",
        "parent_name": config.get("parent_name", "Not registered"),
        "registered_profiles": profile_count,
        "ws_listener": phone_audio_listener.status(),
    }


@router.get("/last-result")
def get_last_result() -> dict:
    """
    Returns the latest prediction result from the ESP32 device.
    The frontend polls this every few seconds to show live status.
    """
    return _last_result if _last_result else {"status": "No data yet — waiting for ESP32 audio."}


@router.post("/register-next-chunk")
def register_next_chunk(
    person_name: str = Form(...),
    role: str = Form("parent"),
) -> dict:
    """
    Arms the system to capture the NEXT audio chunk sent by the ESP32
    and use it as the voice registration profile for this person.
    The frontend calls this, then the user speaks near the ESP32.
    """
    global _register_next_for
    _register_next_for = {"person_name": person_name, "role": role}
    logger.info(f"Waiting for next ESP32 chunk to register as '{person_name}' ({role})")
    return {"armed": True, "person_name": person_name, "message": "Speak near the ESP32 now. The next audio chunk will be registered as your voice profile."}


@router.get("/register-next-chunk/status")
def register_next_chunk_status() -> dict:
    """Returns whether the system is armed and waiting for a registration chunk."""
    return {"armed": bool(_register_next_for), "waiting_for": _register_next_for}

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
    Endpoint for the ESP32 microcontroller to upload 3-second audio chunks.
    Runs the 1D-CNN + LSTM model and triggers a Supabase alert if it's a Threat.
    """
    contents = await file.read()
    config = get_config()
    parent_name = config.get("parent_name", "Parent")

    # ── REGISTRATION INTERCEPT ───────────────────────────────────
    # If armed, use this chunk as a voice registration instead of threat detection
    global _register_next_for
    if _register_next_for:
        reg_name = _register_next_for["person_name"]
        reg_role = _register_next_for["role"]
        _register_next_for = {}  # disarm immediately

        with open(PARENT_PROFILE_PATH, "wb") as f:
            f.write(contents)

        mfcc_matrix = predictor.extract_mfcc_matrix(contents, n_mfcc=20)
        if mfcc_matrix is None:
            logger.error(f"ESP32 registration chunk for '{reg_name}' was too short or silent.")
            return {"registered": False, "error": "Chunk was too short or silent. Try again."}

        # Deactivate previous profiles for this person
        try:
            db.table("registered_voice_profiles").update({"is_active": False}).eq("person_name", reg_name).execute()
        except Exception:
            pass

        db.table("registered_voice_profiles").insert({
            "person_name": reg_name,
            "role": reg_role,
            "dtw_feature_matrix": mfcc_matrix.tolist(),
            "is_active": True,
        }).execute()

        # Update config
        config["parent_name"] = reg_name
        save_config(config)
        logger.info(f"ESP32 chunk registered as voice profile for '{reg_name}'.")

        return {
            "registered": True,
            "person_name": reg_name,
            "role": reg_role,
            "mfcc_shape": list(mfcc_matrix.shape),
            "status": f"Voice profile for {reg_name} saved!",
        }
    # ── END REGISTRATION INTERCEPT ───────────────────────────────

    import librosa
    try:
        y, sr = librosa.load(io.BytesIO(contents), sr=None)
        rms = np.sqrt(np.mean(y**2))
        rms_scaled = rms * 32767.0
        amplitude_db = float(20 * np.log10(rms_scaled) if rms_scaled > 0 else 0.0)
    except Exception:
        amplitude_db = 0.0

    # ── 2. Run the Deep Learning Model ───────────────────────
    class_id, probability = predictor.predict_from_wav_bytes(contents)
    class_id = int(class_id)
    probability = float(probability)
    
    status_msg = "Safe"
    mitigation_msg = None
    is_parent = False

    # ── 3. Parent Voice Verification ─────────────────────────
    # Only check if there's enough volume to be a voice
    if amplitude_db > 45.0:
        # Primary: Check against all Supabase registered profiles
        stored_matrices = get_registered_mfcc_profiles()
        if stored_matrices:
            is_parent = predictor.verify_parent_from_matrix(contents, stored_matrices)
            logger.info(f"Supabase profile verification: {'MATCH' if is_parent else 'no match'}")
        else:
            # Fallback: local WAV file (backward compatibility)
            is_parent = predictor.verify_parent(contents, PARENT_PROFILE_PATH)
            if is_parent:
                logger.info("Parent verified via local WAV fallback.")

    # ── 4. Intensity Override (compensates for overfitted model) ─
    if class_id == 0 and amplitude_db > 80.0:
        class_id = 1
        probability = 0.99
        mitigation_msg = f"Intensity Override: Audio at {amplitude_db:.1f}dB flagged as threat."
        
    # ── 5. Final Decision Logic ───────────────────────────────
    if class_id == 1:
        if is_parent:
            # Parent is speaking loudly — override the threat
            status_msg = f"Safe ({parent_name} speaking — Threat Override)"
            class_id = 0
        else:
            threat_level = "high" if probability >= 0.85 else "moderate"
            status_msg = f"Threat Detected ({threat_level.capitalize()})"
            phone_audio_listener._trigger_supabase_alert(
                intensity_score=probability * 100.0, 
                threat_level=threat_level,
                device_info=device_info
            )
    else:
        if is_parent:
            status_msg = f"Safe ({parent_name} speaking)"
        elif amplitude_db > 70.0:
            mitigation_msg = f"Anti-Fatigue: {amplitude_db:.1f}dB detected but AI confirmed SAFE. Alert suppressed."

    result = {
        "filename": file.filename,
        "class_id": class_id,
        "status": status_msg,
        "probability": f"{probability:.2%}",
        "amplitude_db": round(float(amplitude_db), 2),
        "mitigation_message": mitigation_msg,
        "is_parent": is_parent,
        "device_info": device_info,
    }

    # Store as last result so the frontend can poll /last-result
    global _last_result
    _last_result = result

    return result


@router.post("/register-parent")
async def register_parent_voice(
    file: UploadFile = File(...),
    parent_name: str = Form("Parent"),
    role: str = Form("parent"),
):
    """
    Registers a voice profile by:
      1. Saving the WAV file locally (backward compat)
      2. Extracting an MFCC matrix
      3. Inserting the matrix into Supabase registered_voice_profiles
      4. Updating the local config with the parent name
    """
    import librosa
    contents = await file.read()
    logger.info(f"register-parent: received file '{file.filename}', size={len(contents)} bytes, content_type={file.content_type}")

    # Save WAV locally as fallback
    with open(PARENT_PROFILE_PATH, "wb") as f:
        f.write(contents)

    # Extract MFCC matrix for Supabase storage
    mfcc_matrix = predictor.extract_mfcc_matrix(contents, n_mfcc=20)
    if mfcc_matrix is None:
        return {
            "success": False,
            "error": "Audio was too short or silent. Please record a longer sample (at least 3 seconds).",
        }

    # Convert numpy array to a plain nested list for JSON storage
    mfcc_list = mfcc_matrix.tolist()

    # Deactivate any previous profiles for this person before inserting new one
    try:
        db.table("registered_voice_profiles") \
            .update({"is_active": False}) \
            .eq("person_name", parent_name) \
            .execute()
    except Exception as e:
        logger.warning(f"Could not deactivate old profiles: {e}")

    # Insert new voice profile into Supabase
    try:
        db.table("registered_voice_profiles").insert({
            "person_name": parent_name,
            "role": role,
            "dtw_feature_matrix": mfcc_list,
            "is_active": True,
        }).execute()
        logger.info(f"Voice profile for '{parent_name}' saved to Supabase.")
    except Exception as e:
        logger.error(f"Supabase insert failed: {e}")
        return {
            "success": False,
            "error": f"Database error saving profile: {str(e)}",
        }

    # Update local config
    config = get_config()
    config["parent_name"] = parent_name
    save_config(config)
    
    return {
        "success": True,
        "message": f"Voice profile for '{parent_name}' registered successfully.",
        "parent_name": parent_name,
        "role": role,
        "mfcc_shape": list(mfcc_matrix.shape),
    }


@router.get("/profiles")
def list_voice_profiles() -> dict:
    """
    Returns all active registered voice profiles (without the large MFCC matrix).
    Used by the frontend to show who is registered.
    """
    try:
        result = db.table("registered_voice_profiles") \
            .select("id, person_name, role, is_active, created_at, last_verified") \
            .order("created_at", desc=True) \
            .execute()
        return {"profiles": result.data or []}
    except Exception as e:
        logger.error(f"Failed to list profiles: {e}")
        return {"profiles": [], "error": str(e)}


@router.delete("/profiles/{profile_id}")
def delete_voice_profile(profile_id: str) -> dict:
    """
    Soft-deletes (deactivates) a voice profile by its ID.
    """
    try:
        db.table("registered_voice_profiles") \
            .update({"is_active": False}) \
            .eq("id", profile_id) \
            .execute()
        return {"success": True, "message": f"Profile {profile_id} deactivated."}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/clear-alerts")
async def clear_alerts():
    """
    Deletes all test alert data from the Supabase audio_threat_alerts table.
    """
    try:
        db.table('audio_threat_alerts').delete().neq('sensor_type', 'dummy').execute()
        return {"message": "Test data cleared successfully."}
    except Exception as e:
        return {"error": f"Failed to clear data: {str(e)}"}
