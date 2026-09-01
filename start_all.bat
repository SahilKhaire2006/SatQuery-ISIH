@echo off
echo ========================================================
echo Starting SatQuery Full-Stack Platform
echo ========================================================
echo Launching Backend Server on http://localhost:8000 ...
start "SatQuery Backend (FastAPI)" cmd /k "start_server.bat"

echo Launching Frontend Application on http://localhost:3000 ...
start "SatQuery Frontend (React)" cmd /k "start_frontend.bat"

echo ========================================================
echo Both services launched! Access app at http://localhost:3000
echo ========================================================
