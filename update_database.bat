@echo off
cd /d "%~dp0"
if exist "legal-rag-backend\venv\Scripts\python.exe" (
    set LOCAL_PYTHON=venv\Scripts\python.exe
) else (
    set LOCAL_PYTHON=..\venv\Scripts\python.exe
)

cd /d "%~dp0legal-rag-backend"

echo ======================================================
echo 1/3: Translating missing fields in penal.json to Sinhala...
echo ======================================================
%LOCAL_PYTHON% scripts\translate_db.py

echo.
echo ======================================================
echo 2/3: Converting penal.json to RAG schema...
echo ======================================================
%LOCAL_PYTHON% scripts\convert_penal.py --overwrite

echo.
echo ======================================================
echo 3/3: Rebuilding FAISS vector index...
echo ======================================================
%LOCAL_PYTHON% rebuild_index.py

echo.
echo Database update and index rebuild completed successfully!
pause
