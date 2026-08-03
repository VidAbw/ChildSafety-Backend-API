# app/api/v1/router.py
from fastapi import APIRouter
from app.api.v1.endpoints import alerts

api_v1_router = APIRouter()

# Include telemetry and alert endpoints under /api/v1 prefix
api_v1_router.include_router(alerts.router, tags=["Safety Alert Pipeline (v1)"])
