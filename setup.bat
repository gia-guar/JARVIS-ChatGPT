@echo off

echo Checking for Python installation...
where python >nul 2>nul || (
  echo Python not found. Please install Python 3.8 and add it to the system PATH.
  exit /b 1
)

echo Checking for CUDA installation...
where nvcc >nul 2>nul || (
  echo CUDA not found. Please install CUDA and add it to the system PATH.
  exit /b 1
)

echo Detecting CUDA version...
for /f "delims=" %%i in ('powershell -command "(Get-Content -Path $env:CUDA_PATH\version.txt) -replace '^CUDA Version (.+)$', '$1' -replace '^$', 'unknown'"') do set CUDA_VERSION=%%i
echo Found CUDA version: %CUDA_VERSION%

echo Creating a new virtual environment...
call py -3.8 -m venv jarvis_venv
call .\jarvis_venv\Scripts\activate
call python --version
pause
echo Upgrading pip...
call python -m pip install --upgrade pip

echo Installing required packages...
call pip install -r venv_requirements.txt

echo Overwriting the OpenAI-Whisper packages
move .\whisper_edits\__main__.py, .\whisper_edits\model.py .\jarvis_venv\Lib\site-packages\whisper
rmdir whisper_edits

echo Downloading TTS repository as a ZIP file...
call curl -L -o tts.zip https://github.com/coqui-ai/TTS/archive/refs/heads/dev.zip

echo Extracting TTS repository...
powershell -command "Expand-Archive -Path .\tts.zip -DestinationPath .\temp_tts_folder -Force"

echo Moving test_TTS.py into the TTS repository...
copy test_TTS.py temp_tts_folder\TTS-dev

echo Executing test_TTS.py...
call python temp_tts_folder\TTS-dev\test_TTS.py

echo Cleaning up...
rmdir /s /q temp_tts_folder
del tts.zip

echo Installing PyTorch according to your CUDA version...
set pytorch_skipped = 0
if "%CUDA_VERSION%"=="unknown" (
    echo Unable to determine CUDA version. Recommended version for this project is 11.8 https://developer.nvidia.com/cuda-11-8-0-download-archive.
    echo Once installed, run this command in powershell or command prompt: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pause
    set pytorch_skipped = 1
) else (
    setlocal enabledelayedexpansion
    set "CUDA_MAJOR_VERSION=!CUDA_VERSION:.=!"
    if !CUDA_MAJOR_VERSION! GEQ 12 (
        echo CUDA version is 12 or above, pytorch is not available. Please downgrade for operation.
        echo Here are isntructions: https://github.com/bycloudai/SwapCudaVersionWindows
        echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        echo The recommended version for this project is 11.7 https://developer.nvidia.com/cuda-11-7-0-download-archive.
        echo Once installed, run this command in powershell or command prompt: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
        pause
        set pytorch_skipped = 1
    ) 
    endlocal

    if "%CUDA_VERSION%"=="11.8" (
        echo installing pytorch for CUDA 11.8
        call pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        set pytorch_skipped = 0
    )
    if "%CUDA_VERSION%"=="11.7" (
        echo installing pytorch for CUDA 11.7
        call pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
        set pytorch_skipped = 0
    )
    if pytorch_skipped == 1 (
        echo Not recommended CUDA version, some realeases have support, trying direct download.
        for /f "tokens=* delims=." %%a in ("%CUDA_VERSION%") do (
            set "CUDA_VERSION_NO_DOT=%%a%%b"
        )
        call pip install torch -f https://download.pytorch.org/whl/cu%CUDA_VERSION_NO_DOT%/torch_stable.html
        
        echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        echo if it completed successfully, your installation is complete, otherwise pytorch isntallation could be not found. 
        echo Please install pytorch according to your CUDA version: "%CUDA_VERSION%"
    )
)

echo Preparing to install Vicuna GPT from TroubleChute (https://hub.tcno.co/ai/text-ai/vicuna/)
cd Vicuna
powershell .\vicuna.ps1

echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo Setup complete!
echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo Please setup required api keys in the .evn.txt (remove .txt after finished) file, then run the test.bat file or tests.py to test setup
pause