@echo off
REM Run from the repository root even when launching this file by double click.
cd /d "%~dp0\.."
python -m pip install -r sidechallenge\requirements.txt
if errorlevel 1 (echo Error instalant dependències & pause & exit /b 1)
echo Obrint UPC Energy Intelligence a http://127.0.0.1:8502
start "" "http://127.0.0.1:8502"
python -m uvicorn sidechallenge.service:app --host 127.0.0.1 --port 8502
pause
