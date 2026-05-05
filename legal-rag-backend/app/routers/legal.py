from fastapi import APIRouter
from app.services.supabase_service import get_reporting_contacts

router = APIRouter()

@router.get("/contacts")
async def get_contacts():
    return {"contacts": get_reporting_contacts()}