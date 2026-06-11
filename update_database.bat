@echo off
echo ======================================================
echo 1/3: Translating missing fields in penal.json to Sinhala...
echo ======================================================
cd /d "%~dp0legal-rag-backend"
venv\Scripts\python.exe scripts\translate_db.py

echo.
echo ======================================================
echo 2/3: Converting penal.json to RAG schema...
echo ======================================================
venv\Scripts\python.exe scripts\convert_penal.py --overwrite

echo.
echo ======================================================
echo 3/3: Rebuilding FAISS vector index...
echo ======================================================
venv\Scripts\python.exe rebuild_index.py

echo.
echo Database update and index rebuild completed successfully!
pause
