@echo off
echo Starting Legal RAG Backend...
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    set LOCAL_PYTHON=..\venv\Scripts\python.exe
) else if exist "legal-rag-backend\venv\Scripts\python.exe" (
    set LOCAL_PYTHON=venv\Scripts\python.exe
) else (
    echo Error: Virtual environment (venv) not found. Please run setup first.
    pause
    exit /b 1
)

cd legal-rag-backend
%LOCAL_PYTHON% -m uvicorn app.main:app --reload
