# pyrefly: ignore [missing-import]
from fastapi import APIRouter
try:
    from ..services.supabase_service import get_reporting_contacts
except (ImportError, ValueError):
    from app.services.supabase_service import get_reporting_contacts


router = APIRouter()

@router.get("/contacts")
async def get_contacts():
    return {"contacts": get_reporting_contacts()}