@echo off
REM Start the P5R Generative Social Links inference server (real model).
REM Requires: .wvenv virtualenv with llama-cpp-python (CUDA) and model in models/
REM Run once and keep this window open while P5R is running.

cd /d "%~dp0"

if not exist ".wvenv\Scripts\python.exe" (
    echo ERROR: .wvenv not found. Create it with:
    echo   python -m venv .wvenv
    echo   .wvenv\Scripts\pip install -r requirements.txt
    exit /b 1
)

if not exist "models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" (
    echo ERROR: Model not found at models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
    echo Download with:
    echo   .wvenv\Scripts\python -c "from huggingface_hub import hf_hub_download; hf_hub_download('bartowski/Meta-Llama-3.1-8B-Instruct-GGUF', 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', local_dir='models')"
    exit /b 1
)

echo Starting P5R Gen Social Links server (real LLM)...
.wvenv\Scripts\python main.py
