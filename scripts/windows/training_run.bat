@echo off
REM One-click Windows training runner
REM - Creates venv if missing
REM - Installs requirements
REM - Runs training loop (self-checking)

setlocal enableextensions enabledelayedexpansion
cd /d %~dp0\..

if not exist venv (
  echo [INFO] Creating Python virtual environment...
  py -3 -m venv venv || (
    echo [ERROR] Failed to create venv. Ensure Python is installed. & exit /b 2
  )
)

call venv\Scripts\activate.bat || (
  echo [ERROR] Failed to activate venv. & exit /b 2
)

echo [INFO] Installing requirements...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt || (
  echo [ERROR] Failed to install requirements. & exit /b 2
)

echo [INFO] Starting training loop...
python scripts\training_loop.py

endlocal

