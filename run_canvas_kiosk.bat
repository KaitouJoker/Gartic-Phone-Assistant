@echo off
setlocal
title Gartic Phone Virtual Canvas (Kiosk Mode)
cd /d "%~dp0"

echo ======================================================================
echo  Gartic Phone Virtual Canvas [Kiosk Fullscreen Mode]
echo.
echo  - Top exit button / popups are 100%% suppressed.
echo  - Pure fullscreen canvas from monitor coordinate (0, 0).
echo  - To exit: Press [Alt + F4] on your keyboard.
echo ======================================================================

set "HTML_PATH=%~dp0gartic_phone_canvas.html"

if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    echo Starting Chrome in Kiosk Mode...
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --kiosk "%HTML_PATH%"
    exit /b 0
)
if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    echo Starting Chrome in Kiosk Mode...
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --kiosk "%HTML_PATH%"
    exit /b 0
)
if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" (
    echo Starting Chrome in Kiosk Mode...
    start "" "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" --kiosk "%HTML_PATH%"
    exit /b 0
)
if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    echo Starting Edge in Kiosk Mode...
    start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --kiosk "%HTML_PATH%"
    exit /b 0
)
if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    echo Starting Edge in Kiosk Mode...
    start "" "C:\Program Files\Microsoft\Edge\Application\msedge.exe" --kiosk "%HTML_PATH%"
    exit /b 0
)

echo Starting default browser...
start "" "%HTML_PATH%"
exit /b 0
