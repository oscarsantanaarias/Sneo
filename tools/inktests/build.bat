@echo off
setlocal
set "out=%~dp0build"
if not exist "%out%" mkdir "%out%"
set "vcvars="
set "finder=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%finder%" for /f "usebackq delims=" %%i in (`"%finder%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "vcvars=%%i\VC\Auxiliary\Build\vcvars64.bat"
call "%vcvars%" >nul 2>nul
cd /d "%~dp0..\.."
cl /nologo /EHsc /std:c++17 /O2 /arch:AVX2 /W3 /MT master.cpp /Fo:"%out%\\" /Fe:"%out%\master.exe"
