@echo off
cd /d "%~dp0.."
if not exist runtime mkdir runtime
if exist runtime\nvrtc64_120_0.dll goto blas
set TMPDL=%TEMP%\sneo_nvrtc
if not exist %TMPDL% mkdir %TMPDL%
py -m pip download --no-deps --only-binary=:all: --platform win_amd64 nvidia-cuda-nvrtc-cu12==12.9.86 -d %TMPDL% -q
powershell -NoProfile -Command "Expand-Archive -Force (Get-ChildItem '%TMPDL%\*.whl')[0].FullName '%TMPDL%\x'"
copy /y "%TMPDL%\x\nvidia\cuda_nvrtc\bin\nvrtc64_120_0.dll" runtime\ >nul
copy /y "%TMPDL%\x\nvidia\cuda_nvrtc\bin\nvrtc-builtins64_129.dll" runtime\ >nul
:blas
if exist runtime\cublasLt64_12.dll exit /b 0
set TMPBL=%TEMP%\sneo_cublas
if not exist %TMPBL% mkdir %TMPBL%
py -m pip download --no-deps --only-binary=:all: --platform win_amd64 nvidia-cublas-cu12==12.9.2.10 -d %TMPBL% -q
powershell -NoProfile -Command "Expand-Archive -Force (Get-ChildItem '%TMPBL%\*.whl')[0].FullName '%TMPBL%\x'"
copy /y "%TMPBL%\x\nvidia\cublas\bin\cublasLt64_12.dll" runtime\ >nul
copy /y "%TMPBL%\x\nvidia\cublas\bin\cublas64_12.dll" runtime\ >nul
