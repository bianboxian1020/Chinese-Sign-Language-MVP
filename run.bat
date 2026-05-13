@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Starting Xinyu Suyi...
"venv\Scripts\python.exe" "src\main_gui.py"
pause
