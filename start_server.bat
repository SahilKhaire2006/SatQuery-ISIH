@echo off
echo Starting SatQuery FastAPI Backend Server...
call venv\Scripts\activate.bat
uvicorn api.gateway:app --reload --port 8000
pause
