@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===========================================
echo  Xinyu Suyi - Setup
echo ===========================================
echo.

REM Find system Python
set PYTHON_EXE=
for %%p in (python python3) do (
    where %%p >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%i in ('where %%p') do set PYTHON_EXE=%%i
        goto :found_python
    )
)
:found_python

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python not found in PATH.
    echo Please install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)

echo Using: %PYTHON_EXE%
%PYTHON_EXE% --version
echo.

REM Create venv
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYTHON_EXE% -m venv venv --system-site-packages
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
    echo Done.
) else (
    echo Virtual environment already exists.
)

echo.
echo Installing dependencies...
"venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some packages may not have installed correctly.
    echo If PyAudio fails, try: pip install pipwin ^&^& pipwin install pyaudio
)

echo.
echo Downloading MediaPipe model...
if not exist "assets\models\hand_landmarker.task" (
    powershell -Command "Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task' -OutFile 'assets\models\hand_landmarker.task'"
    if errorlevel 1 (
        echo [WARNING] Model download failed.
        echo URL: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
        echo Save to: assets\models\hand_landmarker.task
    )
) else (
    echo Model already exists.
)

echo.
echo ===========================================
echo  Installation complete!
echo  Run run.bat to start.
echo ===========================================
pause
