@echo off
echo Starting Legal RAG Backend...
cd /d "%~dp0legal-rag-backend"
if not exist venv (
    echo Error: Virtual environment (venv) not found. Please run setup first.
    pause
    exit /b 1
)
venv\Scripts\python.exe -m uvicorn app.main:app --reload
