@echo off
cd /d %~dp0
echo [1/3] Creating Python virtual environment (.venv) ...
py -3.12 -m venv .venv
if errorlevel 1 python -m venv .venv
if not exist .venv\Scripts\activate.bat (
  echo Python was not found. Install Python 3.12 first: winget install -e --id Python.Python.3.12
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
echo [2/3] Upgrading pip ...
python -m pip install --upgrade pip
echo [3/3] Installing mediapipe (includes OpenCV) ...
pip install -r requirements.txt
echo.
echo Done. Next: double-click run_demo.bat (test) or run_hand.bat (webcam).
pause
