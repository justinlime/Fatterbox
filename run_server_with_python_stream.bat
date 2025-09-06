@echo off
setlocal enabledelayedexpansion

echo Welcome to the Openai-Compatible Server for Chatterbox-TTS!
echo .

:: Prompt for hardware
echo.
set /p gpuchoice="Run with cuda, cpu, or apple (mps)? (type 'cuda' or 'cpu' or 'mps' and press enter): "
if "%gpuchoice%"=="" set "gpuchoice=cpu"
set "argsdevice=--device %gpuchoice%"

:: Prompt for lowvram
echo.
set "argslowvram="
set /p vramchoice="Run with low vram mode (cuda only) (y/n): "
if /i "%vramchoice%"=="y" set "argslowvram=--low_vram"

:: Prompt for exaggeration
echo.
set /p exaggeration="Enter exaggeration value (recommended 0.1-2.5; default=0.5): "
if "%exaggeration%"=="" set "exaggeration=0.5"
set "argsexaggeration=--exaggeration %exaggeration%"

:: Prompt for temperature
echo.
set /p temperature="Enter temperature value (recommended 0.4-2.0; default=0.8): "
if "%temperature%"=="" set "temperature=0.8"
set "argstemperature=--temperature %temperature%"

:: Prompt for streaming mode
echo.
set "argsstream="
set /p streamchoice="Enable streaming mode? (y/n): "
if /i "%streamchoice%"=="y" set "argsstream=--stream"

:: Prompt for language code
echo.
set /p languageid="Enter two-letter language code (e.g., en, es, fr): "
if "%languageid%"=="" set "languageid=en"

set "argslanguageid="
if not "%languageid%"=="en" (
    set "argslanguageid=--language_id %languageid%"
)

:: Choose model path based on language
if /i "%languageid%"=="en" (
    set "argsmodel=--model_path %~dp0model\chatterbox_model"
) else (
    set "argsmodel=--model_path %~dp0model\chatterbox_multilingual_model"
)

:: Set minp value
set "argsminp=--min-p 0.1"

:: Activate virtual environment
call "%~dp0venv\Scripts\activate.bat" || (
    echo Failed to activate virtual environment.
    exit /b 1
)

:: Run the server in the current terminal
echo.
echo Starting the TTS server...
echo Using language code: %languageid%
echo Model path: %argsmodel%
echo Access a test webpage by CTRL clicking this link after it loads: http://localhost:5002
echo.

:: Run the Python server
python wingman_chatterbox_openai_server_stream.py !argsmodel! !argsdevice! !argslowvram! !argsexaggeration! !argstemperature! !argsstream! !argsminp! !argslanguageid!
