@echo off
REM One-time setup: creates .venv and installs Jaylights
cd /d "%~dp0"
where python >nul 2>nul || (echo Python not found. Install Python 3.12 from python.org and tick "Add python.exe to PATH". & pause & exit /b 1)
python -m venv .venv || (pause & exit /b 1)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -e ".[loopback]" || (pause & exit /b 1)
echo.
echo  Done! Double-click demo.bat to see it in action.
pause
