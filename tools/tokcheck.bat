@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0.."
if not exist build mkdir build

set MODEL=modelos\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
if not "%~1"=="" set MODEL=%~1

if "%TOK_PYTHON%"=="" set TOK_PYTHON=build\venv_tok\Scripts\python.exe
if not exist "%TOK_PYTHON%" (
    set BASE_PYTHON=python
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set BASE_PYTHON="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    call %%BASE_PYTHON%% -m venv build\venv_tok
    if errorlevel 1 exit /b 1
    build\venv_tok\Scripts\python.exe -I -m pip install -q tokenizers regex jinja2
    if errorlevel 1 exit /b 1
    set TOK_PYTHON=build\venv_tok\Scripts\python.exe
)

cl /nologo /EHsc /std:c++17 /O2 /W3 /MT tools\tokcheck.cpp /Fo:build\ /Fe:build\tokcheck.exe
if errorlevel 1 exit /b 1

"%TOK_PYTHON%" -I tools\tokref.py "%MODEL%" build\tokref.json
if errorlevel 1 exit /b 1

build\tokcheck.exe "%MODEL%" build\tokref.json
