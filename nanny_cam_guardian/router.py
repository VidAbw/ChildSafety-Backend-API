# routers/iot.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time
import shutil
from pathlib import Path
from nanny_cam_guardian.detector.capture import NannyCamStreamer

KNOWN_FACES_DIR = Path(__file__).resolve().parent / "known_faces"

streamer = NannyCamStreamer()
from core.supabase import db

router = APIRouter()


class AlertDetails(BaseModel):
    adult_hand_velocity: float | None = None
    skeleton_distance: float | None = None
    triggered_by: list[str] = []


class AlertSchema(BaseModel):
    user_id: str
    source: str = "nanny_cam"   # default — overridden if called by another component
    type: str                   # 'hazard' | 'fall' | 'abuse_suspected'
    probability: float
    timestamp: str
    details: AlertDetails | None = None


@router.post("/alert")
def create_alert(alert: AlertSchema):
    data = {
        "user_id": alert.user_id,
        "source": alert.source,
        "type": alert.type,
        "probability": alert.probability,
        "timestamp": alert.timestamp,
        "details": alert.details.model_dump() if alert.details else None,
    }
    response = db.table("alerts").insert(data).execute()

    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to save alert")

    return {"status": "success", "id": response.data[0]["id"]}

@router.post("/start")
def start_camera():
    streamer.start()
    return {"status": "success", "message": "Camera started"}

@router.post("/stop")
def stop_camera():
    streamer.stop()
    return {"status": "success", "message": "Camera stopped"}

@router.get("/stream")
def stream_camera():
    from fastapi import Response
    frame = streamer.get_frame()
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    if frame:
        return Response(content=frame, media_type="image/jpeg", headers=headers)
    
    # Return a 1x1 transparent pixel if not ready
    transparent_pixel = b'\x47\x49\x46\x38\x39\x61\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00\x21\xf9\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x44\x01\x00\x3b'
    return Response(content=transparent_pixel, media_type="image/gif", headers=headers)

@router.post("/upload-face")
def upload_known_face(name: str = Form(...), file: UploadFile = File(...)):
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sanitize the name so we don't save dangerous filenames
    safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid name provided")
        
    ext = Path(file.filename).suffix or ".jpg"
    file_path = KNOWN_FACES_DIR / f"{safe_name}{ext}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
        
    return {"status": "success", "message": f"Successfully uploaded face for {safe_name}"}

@router.get("/known-faces")
def get_known_faces():
    if not KNOWN_FACES_DIR.exists():
        return {"faces": []}
    
    faces = []
    for f in KNOWN_FACES_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            faces.append(f.stem)
            
    return {"faces": faces}

@router.delete("/known-face/{name}")
def delete_known_face(name: str):
    if not KNOWN_FACES_DIR.exists():
        raise HTTPException(status_code=404, detail="Directory not found")
        
    safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()
    
    deleted = False
    for f in KNOWN_FACES_DIR.glob(f"{safe_name}.*"):
        try:
            f.unlink()
            deleted = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete {f.name}: {str(e)}")
            
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Face for {safe_name} not found")
        
    return {"status": "success", "message": f"Successfully deleted {safe_name}"}
