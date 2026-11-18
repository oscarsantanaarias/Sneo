@echo off
setlocal
cd /d "%~dp0"
set "vcvars="
set "finder=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%finder%" for /f "usebackq delims=" %%i in (`"%finder%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "vcvars=%%i\VC\Auxiliary\Build\vcvars64.bat"
if not defined vcvars set "vcvars=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%vcvars%" (
    echo Visual Studio with the C++ tools was not found.
    echo Open Visual Studio Installer, click Modify, check Desktop development with C++ and run build.bat again.
    echo Or download the ready to run zip from https://github.com/oscarsantanaarias/Sneo/releases
    goto done
)
call "%vcvars%" >nul 2>nul
if not exist build mkdir build

echo Building the engine tools...
for %%t in (fetchdraft draftbuild repack) do (
    cl /nologo /EHsc /std:c++17 /O2 /arch:AVX2 /W3 /MT tools\%%t.cpp /Fo:build\ /Fe:build\%%t.exe || goto failed
    copy /y build\%%t.exe tools\%%t.exe >nul || goto busy
)

echo Building the engine...
cl /nologo /EHsc /std:c++17 /O2 /arch:AVX2 /W3 /MT master.cpp /Fo:build\ /Fe:build\master.exe || goto failed
cl /nologo /EHsc /std:c++17 /O2 /arch:AVX2 /W3 /MT worker.cpp /Fo:build\ /Fe:build\worker.exe || goto failed
copy /y build\master.exe master.exe >nul || goto busy
copy /y build\worker.exe worker.exe >nul || goto busy
echo Done: master.exe, worker.exe and the tools are up to date.
goto done

:busy
echo A program is still running from this folder: close Sneo (master.exe) and run build.bat again.
goto done

:failed
echo The build failed, the messages above say why.

:done
if /i not "%~1"=="nopause" pause
