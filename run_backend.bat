@echo off
echo Starting Legal RAG Backend...
cd /d "%~dp0"
if not exist venv (
    echo Error: Virtual environment (venv) not found at root. Please run setup first.
    pause
    exit /b 1
)
cd legal-rag-backend
..\venv\Scripts\python.exe -m uvicorn app.main:app --reload
