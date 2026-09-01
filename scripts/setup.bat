@echo off
echo Setting up SatQuery project...

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
pip install -r requirements.txt

REM Create data directories
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist data\models mkdir data\models
if not exist logs mkdir logs

REM Copy environment file
if not exist .env (
    copy .env.example .env
    echo Created .env file. Please update with your configuration.
)

echo Setup complete!
echo.
echo To start the API server:
echo   python main.py
echo.
echo To start the CLI:
echo   python cli.py

pause
