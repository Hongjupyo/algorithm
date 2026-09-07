@echo off
cd /d "%~dp0"
echo.
echo Starting missing person assessment server...
echo OpenAI settings: .env
echo The browser will open after the server is ready.
echo Close this window to stop the local server.
echo.
"C:\Users\NSU\miniconda3\envs\py311\python.exe" server.py
