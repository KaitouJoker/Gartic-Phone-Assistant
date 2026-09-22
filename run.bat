@echo off
title Gartic Phone Assistant
cd /d "%~dp0"

echo ========================================================
echo   Gartic Phone Assistant Starting...
echo ========================================================
echo.

python -W ignore line_draw.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================================
    echo   An error occurred. Check the error message above.
    echo ========================================================
    pause
)
