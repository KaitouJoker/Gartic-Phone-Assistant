@echo off
setlocal
title Gartic Phone Virtual Canvas (Kiosk Mode)
cd /d "%~dp0"

echo ======================================================================
echo  Gartic Phone Virtual Canvas [Kiosk Fullscreen Mode]
echo.
echo  - Top exit button / popups are 100%% suppressed.
echo  - Pure fullscreen canvas from monitor coordinate (0, 0).
echo  - Isolated Chrome Profile: External app links will NOT overwrite!
echo  - To exit: Press [Alt + F4] on your keyboard.
echo ======================================================================

set "HTML_PATH=%~dp0gartic_phone_canvas.html"
set "PROFILE_DIR=%TEMP%\gartic_canvas_profile"

if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    echo Starting Chrome in Kiosk Mode (Isolated Profile)...
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --user-data-dir="%PROFILE_DIR%" --no-first-run --no-default-browser-check --kiosk "%HTML_PATH%"
    exit /b 0
)
if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    echo Starting Chrome in Kiosk Mode (Isolated Profile)...
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --user-data-dir="%PROFILE_DIR%" --no-first-run --no-default-browser-check --kiosk "%HTML_PATH%"
    exit /b 0
)
if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" (
    echo Starting Chrome in Kiosk Mode (Isolated Profile)...
    start "" "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" --user-data-dir="%PROFILE_DIR%" --no-first-run --no-default-browser-check --kiosk "%HTML_PATH%"
    exit /b 0
)
if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    echo Starting Edge in Kiosk Mode (Isolated Profile)...
    start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --user-data-dir="%PROFILE_DIR%" --no-first-run --no-default-browser-check --kiosk "%HTML_PATH%"
    exit /b 0
)
if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    echo Starting Edge in Kiosk Mode (Isolated Profile)...
    start "" "C:\Program Files\Microsoft\Edge\Application\msedge.exe" --user-data-dir="%PROFILE_DIR%" --no-first-run --no-default-browser-check --kiosk "%HTML_PATH%"
    exit /b 0
)

echo Starting default browser...
start "" "%HTML_PATH%"
exit /b 0
