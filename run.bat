@echo off
REM 心语速译 (Xinyu Suyi) — 启动脚本
REM Activates the venv and launches the GUI application.

cd /d "%~dp0"

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please run: setup.bat
    pause
    exit /b 1
)

REM Launch the application
echo Starting 心语速译...
"venv\Scripts\python.exe" "src\main_gui.py"
pause
