@echo off
REM Start the server in MOCK mode — no model loaded, returns canned responses.
REM Used for mod development and E2E testing without a GPU.

cd /d "%~dp0"

if not exist ".wvenv\Scripts\python.exe" (
    echo ERROR: .wvenv not found. Run: python -m venv .wvenv ^&^& .wvenv\Scripts\pip install -r requirements.txt
    exit /b 1
)

echo Starting P5R Gen Social Links server (MOCK mode)...
set MOCK_LLM=1
.wvenv\Scripts\python main.py
