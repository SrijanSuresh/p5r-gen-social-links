@echo off
REM Start the P5R Generative Social Links inference server (real model).
REM
REM Launches the FastAPI app, which in turn spawns llama-server.exe as a child
REM process and waits for the weights to load. Run once and keep this window open
REM while P5R is running; closing it stops both processes.
REM
REM Requires: .wvenv virtualenv, vendor\llama-server.exe, and the GGUF in models\.

cd /d "%~dp0"

if not exist ".wvenv\Scripts\python.exe" (
    echo ERROR: .wvenv not found. Create it with:
    echo   python -m venv .wvenv
    echo   .wvenv\Scripts\pip install -r requirements.txt
    exit /b 1
)

if not exist "vendor\llama-server.exe" (
    echo ERROR: llama-server.exe not found in vendor\
    echo Fetch the prebuilt CUDA binaries with:
    echo   powershell -ExecutionPolicy Bypass -File ..\scripts\fetch-llama-server.ps1
    exit /b 1
)

if not exist "models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" (
    echo ERROR: Model not found at models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
    echo Download with:
    echo   .wvenv\Scripts\python -c "from huggingface_hub import hf_hub_download; hf_hub_download('bartowski/Meta-Llama-3.1-8B-Instruct-GGUF', 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', local_dir='models')"
    exit /b 1
)

echo Starting P5R Gen Social Links server (real LLM)...
echo   API           http://127.0.0.1:8765
echo   llama-server  http://127.0.0.1:8766  (child process)
echo   child log     logs\llama-server.log
echo.
echo First start loads ~4.9 GB into VRAM and can take a minute or two.
echo.
.wvenv\Scripts\python main.py
