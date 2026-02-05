# routers/iot.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.supabase import db

router = APIRouter()

# Define the data shape (Validation)
class AlertSchema(BaseModel):
    risk_level: str
    message: str

@router.post("/alert")
def create_alert(alert: AlertSchema):
    # Save to Supabase
    data = {"risk_level": alert.risk_level, "message": alert.message}
    response = db.table("alerts").insert(data).execute()
    
    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to save alert")
        
    return {"status": "success", "id": response.data[0]['id']}