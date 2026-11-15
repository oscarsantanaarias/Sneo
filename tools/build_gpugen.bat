@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0.."
if not exist build mkdir build
cl /nologo /EHsc /std:c++17 /O2 /arch:AVX2 /MT /W3 tools\gpugen.cpp /Fo:build\ /Fe:gpugen.exe
if errorlevel 1 exit /b 1
