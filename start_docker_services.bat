@echo off
echo ========================================================
echo Starting PostgreSQL & Redis via Docker Compose...
echo ========================================================
docker compose up -d db redis
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: Docker Desktop daemon is starting up or not running.
    echo Please make sure Docker Desktop is open and running, then re-run this script.
) else (
    echo.
    echo SUCCESS: PostgreSQL (port 5432) & Redis (port 6379) are now running!
)
pause
