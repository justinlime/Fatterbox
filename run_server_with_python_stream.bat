@echo off 
setlocal enabledelayedexpansion

echo Welcome to the Openai-Compatible Server for Chatterbox-TTS!
echo .

:: Prompt for hardware
echo.
set /p gpu_choice="Run with cuda, cpu, or apple (mps)? (type 'cuda' or 'cpu' or 'mps' and press enter): "
if "%gpu_choice%"=="" set "gpu_choice=cpu"
set "args_device=--device %gpu_choice%"

:: Prompt for lowvram
echo.
set "args_lowvram="
set /p vram_choice="Run with low vram mode (cuda only) (y/n): "
if /i "%vram_choice%"=="y" set "args_lowvram=--low_vram"

:: Prompt for exaggeration
echo.
set /p exaggeration="Enter exaggeration value (recommended 0.1-2.5; default=0.5): "
if "%exaggeration%"=="" set "exaggeration=0.5"
set "args_exaggeration=--exaggeration %exaggeration%"

:: Prompt for temperature
echo.
set /p temperature="Enter temperature value (recommended 0.4-2.0; default=0.8): "
if "%temperature%"=="" set "temperature=0.8"
set "args_temperature=--temperature %temperature%"

:: Prompt for streaming mode
echo.
set "args_stream="
set /p stream_choice="Enable streaming mode? (y/n): "
if /i "%stream_choice%"=="y" set "args_stream=--stream"

:: Set model path argument (relative to this script's directory)
set "args_model=--model_path %~dp0model\chatterbox_model"

:: Set min_p value
set "args_min_p=--min-p 0.1"

:: Activate virtual environment
call "%~dp0venv\Scripts\activate.bat" || (
    echo Failed to activate virtual environment.
    exit /b 1
)

:: Run the server in the current terminal
echo.
echo Starting the TTS server...
echo Access a test webpage by CTRL clicking this link after it loads: http://localhost:5002
echo.

:: Run the Python server
python wingman_chatterbox_openai_server_stream.py !args_model! !args_device! !args_lowvram! !args_exaggeration! !args_temperature! !args_stream! !args_min_p!
