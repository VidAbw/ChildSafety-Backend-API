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
AI_THREAT_THRESHOLD = 0.80

router = APIRouter()

# In-memory store of the latest ESP32 prediction (so the frontend can poll it)
_last_result: dict = {}
_last_verified_speaker: dict = {"name": None, "role": None, "timestamp": 0}

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
    Returns a list of profile dicts with MFCC 2D arrays ready for DTW comparison.
    """
    try:
        result = db.table("registered_voice_profiles") \
            .select("id, dtw_feature_matrix, person_name, role") \
            .eq("is_active", True) \
            .execute()

        rows = result.data or []
        profiles = []
        for row in rows:
            matrix = row.get("dtw_feature_matrix")
            if matrix is not None:
                profiles.append({
                    "id": row.get("id"),
                    "matrix": matrix,
                    "person_name": row.get("person_name", "Parent"),
                    "role": row.get("role", "Parent")
                })
        logger.info(f"Loaded {len(profiles)} active voice profile(s) from Supabase.")
        return profiles
    except Exception as e:
        logger.warning(f"Could not load profiles from Supabase: {e}")
        return []

# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────
@router.get("/status")
def get_audio_listener_status(user_email: str = None) -> dict:
    config = get_config()
    import time
    now_ts = time.time()
    time_since_speech = now_ts - _last_verified_speaker.get("timestamp", 0)
    is_active_presence = (_last_verified_speaker.get("name") is not None) and (time_since_speech <= 30.0)

    try:
        query = db.table("registered_voice_profiles") \
            .select("id, person_name, role, is_active, last_verified, user_email") \
            .eq("is_active", True)
        if user_email and user_email.strip():
            try:
                filtered_query = query.eq("user_email", user_email.strip().lower()).execute()
                profile_list = filtered_query.data or []
            except Exception:
                profiles = query.execute()
                profile_list = profiles.data or []
        else:
            profiles = query.execute()
            profile_list = profiles.data or []
        profile_count = len(profile_list)
    except Exception:
        try:
            profiles = db.table("registered_voice_profiles") \
                .select("id, person_name, role, is_active, last_verified") \
                .eq("is_active", True) \
                .execute()
            profile_list = profiles.data or []
            profile_count = len(profile_list)
        except Exception:
            profile_list = []
            profile_count = 0

    # Annotate profiles with live nearby status
    for p in profile_list:
        p_name = p.get("person_name")
        p["is_currently_nearby"] = bool(
            is_active_presence and p_name == _last_verified_speaker.get("name")
        )

    active_spk = _last_verified_speaker.get("name") if is_active_presence else None
    spk_role = _last_verified_speaker.get("role") if is_active_presence else None
    presence_text = "Active Nearby" if is_active_presence else "Monitoring Area"

    return {
        "backend": "online",
        "parent_name": config.get("parent_name", "Not registered"),
        "registered_profiles": profile_count,
        "active_profiles": profile_list,
        "latest_presence": presence_text,
        "active_speaker": active_spk,
        "speaker_role": spk_role,
        "last_seen": _last_result.get("last_seen"),
        "last_status": _last_result.get("status", "System Active"),
        "ws_listener": {"disabled": True, "message": "Phone audio listener has been disabled."},
    }


@router.get("/last-result")
def get_last_result() -> dict:
    """
    Returns the latest prediction result from the ESP32 device.
    The frontend polls this every few seconds to show live status.
    """
    import time
    now_ts = time.time()
    time_since_speech = now_ts - _last_verified_speaker.get("timestamp", 0)
    is_active_presence = (_last_verified_speaker.get("name") is not None) and (time_since_speech <= 30.0)

    if not _last_result:
        return {"status": "No data yet — waiting for ESP32 audio."}

    res = dict(_last_result)
    if not is_active_presence:
        res["active_speaker"] = None
        res["speaker_role"] = None
        res["presence_status"] = "Monitoring Area"

    return res


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
    return {
        "message": "Phone audio listener is disabled.",
        "status": {"disabled": True},
    }


@router.post("/stop")
async def stop_audio_listener() -> dict:
    return {
        "message": "Phone audio listener is disabled.",
        "status": {"disabled": True},
    }


@router.post("/upload-chunk")
async def upload_audio_chunk(
    file: UploadFile = File(...),
    device_info: str = Form("unknown"),
    vision_tier: int = Form(0),
):
    """
    Endpoint for the ESP32 microcontroller to upload 3-second audio chunks.
    Runs the 1D-CNN + LSTM threat model, calibrated INMP441 SPL analysis,
    64-D standardized biometric caregiver verification, and pediatric distress band arbitration.
    """
    contents = await file.read()
    config = get_config()
    parent_name = config.get("parent_name", "Parent")

    # ── REGISTRATION INTERCEPT ───────────────────────────────────
    # If armed, use this chunk as a voice registration instead of threat detection
    global _register_next_for, _last_result, _last_verified_speaker
    if _register_next_for:
        reg_name = _register_next_for["person_name"]
        reg_role = _register_next_for["role"]
        _register_next_for = {}  # disarm immediately

        mfcc_matrix, vad_err = predictor.extract_mfcc_matrix(contents, n_mfcc=20, require_vad=True)
        if mfcc_matrix is None:
            logger.error(f"ESP32 registration chunk for '{reg_name}' failed VAD: {vad_err}")
            return {"registered": False, "error": vad_err or "No voice detected. Please speak clearly near the microphone."}

        with open(PARENT_PROFILE_PATH, "wb") as f:
            f.write(contents)

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

    try:
        y = predictor.decode_audio(contents, target_sr=22050)
        if y is not None and len(y) > 0:
            rms_float, L_dBFS, L_SPL = predictor.compute_calibrated_spl(y)
        else:
            rms_float, L_dBFS, L_SPL = 0.0, -180.0, 0.0
    except Exception:
        rms_float, L_dBFS, L_SPL = 0.0, -180.0, 0.0

    amplitude_db = L_SPL
    logger.info(f"Audio chunk level: L_dBFS={L_dBFS:.2f} dBFS, L_SPL={L_SPL:.2f} dB SPL (calibrated INMP441)")

    # ── 2. Run the Deep Learning Model ───────────────────────
    class_id, probability = predictor.predict_from_wav_bytes(contents)
    class_id = int(class_id)
    probability = float(probability)

    # ── Compute High-Frequency Pediatric Distress Band Energy Ratio ──
    # R_scream = sum(P[2500 Hz : 5000 Hz]) / sum(P[20 Hz : 8000 Hz])
    R_scream = predictor.compute_scream_ratio(y, sr=22050) if y is not None else 0.0
    logger.info(f"Pediatric distress band energy ratio R_scream={R_scream:.4f}")

    status_msg = "Safe"
    mitigation_msg = None
    is_parent = False
    dtw_dist = 1.0

    # ── 3. Parent Voice Verification ─────────────────────────
    matched_profile = None
    import time
    now_ts = time.time()

    # Caregiver voice match evaluation threshold: L_SPL >= 65.0 dB SPL (-55.0 dBFS) or class_id == 1
    if L_SPL >= 65.0 or class_id == 1:
        stored_profiles = get_registered_mfcc_profiles()
        if stored_profiles:
            is_parent, matched_profile, dtw_dist = predictor.verify_parent_from_matrix(contents, stored_profiles)
            if is_parent and matched_profile:
                _last_verified_speaker = {
                    "name": matched_profile.get("name"),
                    "role": matched_profile.get("role", "Parent"),
                    "timestamp": now_ts,
                }
                if matched_profile.get("id"):
                    try:
                        from datetime import datetime, timezone
                        now_iso = datetime.now(timezone.utc).isoformat()
                        db.table("registered_voice_profiles").update({"last_verified": now_iso}).eq("id", matched_profile["id"]).execute()
                    except Exception as e:
                        logger.debug(f"Could not update last_verified in Supabase: {e}")
            logger.info(f"Supabase profile verification: {'MATCH (' + str(matched_profile.get('name') if matched_profile else '') + ')' if is_parent else 'no match'}, dist={dtw_dist:.4f}")
        else:
            # Fallback: local WAV file (backward compatibility)
            is_parent = predictor.verify_parent(contents, PARENT_PROFILE_PATH)
            if is_parent:
                dtw_dist = 0.05
                matched_profile = {"name": parent_name, "role": "Parent", "id": None}
                _last_verified_speaker = {"name": parent_name, "role": "Parent", "timestamp": now_ts}
                logger.info("Parent verified via local WAV fallback.")

    # Caregiver Voice Match: Dist <= 0.125 and L_SPL >= 65.0 dB SPL (-55.0 dBFS)
    match_caregiver = bool(is_parent and dtw_dist <= 0.125 and L_SPL >= 65.0)

    # ── Presence Memory Hysteresis (15-Second Grace Window for speech continuation) ──
    time_since_last_speech = now_ts - _last_verified_speaker.get("timestamp", 0)
    if not match_caregiver and time_since_last_speech < 15.0 and L_SPL >= 65.0 and class_id == 0:
        if _last_verified_speaker.get("name"):
            match_caregiver = True
            is_parent = True
            matched_profile = {
                "name": _last_verified_speaker.get("name"),
                "role": _last_verified_speaker.get("role", "Parent"),
                "id": None,
            }
            logger.info(f"Presence Hysteresis: Retained presence for '{matched_profile['name']}' (last heard {time_since_last_speech:.1f}s ago).")

    speaker_name = (matched_profile.get("name") if matched_profile else None) or parent_name
    speaker_role = (matched_profile.get("role") if matched_profile else None) or "Parent"
    speaker_display = f"{speaker_name} ({speaker_role})" if matched_profile else speaker_name

    # ── 4. Intensity Override (80.0 dB SPL corresponds to -40.0 dBFS) ──────
    if class_id == 0 and L_SPL > 80.0:
        class_id = 1
        probability = 0.99
        mitigation_msg = f"Intensity Override: Audio at {L_SPL:.1f} dB SPL ({L_dBFS:.1f} dBFS) flagged as threat."

    # ── 5. Context-Safe Caregiver Arbitration with Pediatric Scream Band Veto (Eq. 4 Fix) ──
    scream_veto = (R_scream >= 0.22)
    vision_veto = (vision_tier > 0)

    if class_id == 1:
        if match_caregiver:
            # Caregiver voice match (Dist <= 0.125 and L_SPL >= 65.0) NEVER unconditionally silences an alert
            # if either (a) R_scream >= 0.22 OR (b) vision_tier > 0
            if scream_veto:
                threat_level = "high"
                status_msg = f"Threat Detected (Pediatric Distress Veto: R_scream={R_scream:.3f} >= 0.22 with {speaker_display})"
                mitigation_msg = f"Caregiver Override Vetoed: High-frequency child scream band energy ratio {R_scream:.3f} >= 0.22 detected despite caregiver voice match."
                phone_audio_listener._trigger_supabase_alert(
                    intensity_score=probability * 100.0,
                    threat_level=threat_level,
                    device_info=device_info
                )
            elif vision_veto:
                threat_level = "high"
                status_msg = f"Threat Detected (Vision Hazard Veto: vision_tier={vision_tier} with {speaker_display})"
                mitigation_msg = f"Caregiver Override Vetoed: Concurrent visual hazard active (tier {vision_tier}) despite caregiver voice match."
                phone_audio_listener._trigger_supabase_alert(
                    intensity_score=probability * 100.0,
                    threat_level=threat_level,
                    device_info=device_info
                )
            else:
                # MatchParent == True AND R_scream < 0.22 AND vision_tier == 0 -> Contextualized_Safe (Push_Suppressed)
                status_msg = "Contextualized_Safe (Push_Suppressed)"
                class_id = 0
                mitigation_msg = f"Anti-Fatigue: Authorized caregiver ({speaker_display}, Dist={dtw_dist:.3f} <= 0.125, L_SPL={L_SPL:.1f} dB SPL) verified with no pediatric scream (R_scream={R_scream:.3f} < 0.22) and vision_tier=0. Push suppressed."
        else:
            threat_level = "high" if probability >= 0.85 else "moderate"
            status_msg = f"Threat Detected ({threat_level.capitalize()})"
            phone_audio_listener._trigger_supabase_alert(
                intensity_score=probability * 100.0,
                threat_level=threat_level,
                device_info=device_info
            )
    else:
        # class_id == 0
        if match_caregiver:
            if scream_veto:
                class_id = 1
                threat_level = "high"
                status_msg = f"Threat Detected (Pediatric Distress Veto: R_scream={R_scream:.3f} >= 0.22 with {speaker_display})"
                mitigation_msg = f"Pediatric Distress Alert: Child scream detected in 2.5-5.0 kHz band (R_scream={R_scream:.3f}) despite caregiver presence."
                phone_audio_listener._trigger_supabase_alert(
                    intensity_score=95.0,
                    threat_level=threat_level,
                    device_info=device_info
                )
            elif vision_veto:
                class_id = 1
                threat_level = "high"
                status_msg = f"Threat Detected (Vision Hazard Veto: vision_tier={vision_tier} with {speaker_display})"
                mitigation_msg = f"Visual Hazard Alert: Active visual hazard (tier {vision_tier}) during caregiver presence."
                phone_audio_listener._trigger_supabase_alert(
                    intensity_score=90.0,
                    threat_level=threat_level,
                    device_info=device_info
                )
            else:
                status_msg = "Contextualized_Safe (Push_Suppressed)"
                mitigation_msg = f"Anti-Fatigue: Authorized caregiver ({speaker_display}, Dist={dtw_dist:.3f} <= 0.125, L_SPL={L_SPL:.1f} dB SPL) verified with no pediatric scream (R_scream={R_scream:.3f} < 0.22) and vision_tier=0. Push suppressed."
        elif L_SPL >= 75.0:
            if scream_veto:
                class_id = 1
                threat_level = "high"
                status_msg = f"Threat Detected (Pediatric Distress: R_scream={R_scream:.3f} >= 0.22)"
                phone_audio_listener._trigger_supabase_alert(
                    intensity_score=90.0,
                    threat_level=threat_level,
                    device_info=device_info
                )
            else:
                mitigation_msg = f"Anti-Fatigue: {L_SPL:.1f} dB SPL detected but AI confirmed SAFE. Alert suppressed."
                status_msg = "Safe (Confirmed Safe)"
        elif L_SPL < 65.0:
            status_msg = "Safe (Quiet / Normal)"
        else:
            status_msg = "Safe (Normal)"

    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    
    # Active Presence Window (30 seconds)
    if is_parent and matched_profile:
        time_since_last_speech = 0.0
        is_active_presence = True
        current_active_speaker = speaker_name
        current_speaker_role = speaker_role
        presence_status = "Active Nearby"
    else:
        time_since_last_speech = now_ts - _last_verified_speaker.get("timestamp", 0)
        is_active_presence = (_last_verified_speaker.get("name") is not None) and (time_since_last_speech <= 30.0)
        current_active_speaker = _last_verified_speaker.get("name") if is_active_presence else None
        current_speaker_role = _last_verified_speaker.get("role") if is_active_presence else None
        presence_status = "Active Nearby" if is_active_presence else "Monitoring Area"

    result = {
        "filename": file.filename,
        "class_id": class_id,
        "status": status_msg,
        "probability": f"{probability:.2%}",
        "L_dBFS": round(float(L_dBFS), 2),
        "L_SPL": round(float(L_SPL), 2),
        "amplitude_db": round(float(L_SPL), 2),
        "R_scream": round(float(R_scream), 4),
        "vision_tier": int(vision_tier),
        "mitigation_message": mitigation_msg,
        "is_parent": is_parent,
        "match_parent": match_caregiver,
        "biometric_distance": round(float(dtw_dist), 4),
        "active_speaker": current_active_speaker,
        "speaker_role": current_speaker_role,
        "presence_status": presence_status,
        "last_seen": now_iso if is_parent else (_last_result.get("last_seen") if _last_result else None),
        "device_info": device_info,
    }

    # Store as last result so the frontend can poll /last-result
    _last_result = result

    return result


@router.post("/check-environment")
async def check_environment(file: UploadFile = File(...)):
    """
    Step 1 helper: Checks microphone input and ambient background noise level.
    """
    contents = await file.read()
    res = predictor.check_acoustic_environment(contents)
    return res


@router.post("/validate-phrase-sample")
async def validate_phrase_sample(file: UploadFile = File(...)):
    """
    Step 2 helper: Validates phrase recitation clarity, SNR, and vocal harmonics.
    """
    contents = await file.read()
    res = predictor.validate_phrase_sample(contents)
    return res


@router.post("/register-parent")
@router.post("/register-voice")
async def register_parent_voice(
    file: UploadFile = File(...),
    parent_name: str = Form("Parent"),
    role: str = Form("parent"),
    user_email: str = Form(None),
):
    """
    Registers an authorized voice profile by:
      1. Validating speech presence via multi-stage Voice Activity Detection (VAD)
      2. Extracting a 64-dimensional Text-Independent Biometric Voiceprint Vector
      3. Inserting the biometric vector into Supabase registered_voice_profiles with user_email
      4. Updating local config and saving reference WAV file
    """
    contents = await file.read()
    logger.info(f"register-parent: received file '{file.filename}', size={len(contents)} bytes, content_type={file.content_type}, user_email={user_email}")

    # Decode and validate voice features via VAD
    y = predictor.decode_audio(contents, target_sr=22050)
    if y is None or len(y) < 100:
        return {"success": False, "error": "Audio could not be decoded or was empty."}

    is_valid, y_speech, vad_error = predictor.validate_and_extract_voice(y, sr=22050)
    if not is_valid or y_speech is None:
        logger.warning(f"Voice registration rejected for '{parent_name}': {vad_error}")
        return {
            "success": False,
            "error": vad_error or "No voice detected. Please speak clearly into the microphone for 3 to 5 seconds.",
        }

    # Extract 64-dimensional Biometric Vocal Tract Voiceprint Vector
    biometric_vec = predictor.extract_speaker_biometric_vector(y_speech, sr=22050)
    biometric_list = biometric_vec.tolist()

    # Save WAV locally as fallback only after voice validation succeeds
    with open(PARENT_PROFILE_PATH, "wb") as f:
        f.write(contents)

    # Deactivate any previous profiles for this person under this email before inserting new one
    try:
        query = db.table("registered_voice_profiles").update({"is_active": False}).eq("person_name", parent_name)
        if user_email and user_email.strip():
            try:
                query.eq("user_email", user_email.strip().lower()).execute()
            except Exception:
                query.execute()
        else:
            query.execute()
    except Exception as e:
        logger.warning(f"Could not deactivate old profiles: {e}")

    # Insert new voice profile into Supabase
    profile_data = {
        "person_name": parent_name,
        "role": role,
        "dtw_feature_matrix": biometric_list,
        "is_active": True,
    }
    if user_email and user_email.strip():
        profile_data["user_email"] = user_email.strip().lower()

    try:
        db.table("registered_voice_profiles").insert(profile_data).execute()
        logger.info(f"Biometric voice profile for '{parent_name}' (user: {user_email}) saved to Supabase.")
    except Exception as e:
        logger.warning(f"Insert with user_email failed: {e}. Retrying without user_email...")
        try:
            db.table("registered_voice_profiles").insert({
                "person_name": parent_name,
                "role": role,
                "dtw_feature_matrix": biometric_list,
                "is_active": True,
            }).execute()
            logger.info(f"Biometric voice profile for '{parent_name}' saved to Supabase (fallback).")
        except Exception as e2:
            logger.error(f"Supabase insert failed: {e2}")
            return {
                "success": False,
                "error": f"Database error saving profile: {str(e2)}",
            }

    # Update local config
    config = get_config()
    config["parent_name"] = parent_name
    save_config(config)
    
    return {
        "success": True,
        "message": f"Biometric voice profile for '{parent_name}' registered successfully.",
        "parent_name": parent_name,
        "role": role,
        "user_email": user_email,
        "vector_dims": len(biometric_list),
    }


@router.get("/profiles")
def list_voice_profiles(user_email: str = None) -> dict:
    """
    Returns registered voice profiles (without the large MFCC matrix).
    If user_email is provided, filters by user_email where available.
    """
    try:
        if user_email and user_email.strip():
            try:
                result = db.table("registered_voice_profiles") \
                    .select("id, person_name, role, is_active, created_at, last_verified, user_email") \
                    .eq("user_email", user_email.strip().lower()) \
                    .order("created_at", desc=True) \
                    .execute()
                if result.data is not None:
                    return {"profiles": result.data or []}
            except Exception as fe:
                logger.warning(f"Filter by user_email in Supabase failed ({fe}), fetching all profiles...")

        result = db.table("registered_voice_profiles") \
            .select("id, person_name, role, is_active, created_at, last_verified, user_email") \
            .order("created_at", desc=True) \
            .execute()
        return {"profiles": result.data or []}
    except Exception as e:
        try:
            result = db.table("registered_voice_profiles") \
                .select("id, person_name, role, is_active, created_at, last_verified") \
                .order("created_at", desc=True) \
                .execute()
            return {"profiles": result.data or []}
        except Exception as e2:
            logger.error(f"Failed to list profiles: {e2}")
            return {"profiles": [], "error": str(e2)}


@router.delete("/profiles/{profile_id}")
def delete_voice_profile(profile_id: str, permanent: bool = True) -> dict:
    """
    Deletes or deactivates a voice profile by its ID.
    If permanent is True, removes the record from the database.
    """
    try:
        if permanent:
            db.table("registered_voice_profiles") \
                .delete() \
                .eq("id", profile_id) \
                .execute()
            return {"success": True, "message": f"Profile {profile_id} permanently deleted."}
        else:
            db.table("registered_voice_profiles") \
                .update({"is_active": False}) \
                .eq("id", profile_id) \
                .execute()
            return {"success": True, "message": f"Profile {profile_id} deactivated."}
    except Exception as e:
        # Fallback to soft delete
        try:
            db.table("registered_voice_profiles") \
                .update({"is_active": False}) \
                .eq("id", profile_id) \
                .execute()
            return {"success": True, "message": f"Profile {profile_id} deactivated."}
        except Exception as e2:
            return {"success": False, "error": str(e2)}


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
