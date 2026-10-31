@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
python hand_sender.py --demo
pause
