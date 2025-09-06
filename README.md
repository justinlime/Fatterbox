# Wingman-Chatterbox-Openai-Server
Openai-compatible Text to Speech Server for Chatterbox TTS

## Installation
1. Tested with python 3.11.7
2. Download the code of this repository (https://github.com/teddybear082/Wingman-Chatterbox-Openai-Server/archive/refs/heads/main.zip)
3. create a virtual environment called venv (python -m venv venv)
4. activate virtual environment (./venv/scripts/activate)
5. pip install either cpu_requirements.txt (should work for cpu or apple mps generations) or cuda_requirements.txt (should work for nvidia gpus) (pip install -r cuda_requirements.txt)
6. Run python wingman_chatterbox_openai_server_stream.py followed by any command line arguments (can see arguments with --help), or run run_server_with_python_stream.bat on windows
7. A simple web page to test generations will be available at http://localhost:5002 after server is running, and openai compatible endpoint will be available at http://localhost:5002/v1
