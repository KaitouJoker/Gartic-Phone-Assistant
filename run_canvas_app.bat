@echo off
setlocal
title Gartic Phone Virtual Canvas (App Mode)
cd /d "%~dp0"

echo ======================================================================
echo  Gartic Phone Virtual Canvas [App Mode]
echo  - Runs in standalone window without browser address bar or tabs.
echo ======================================================================

set "HTML_PATH=%~dp0gartic_phone_canvas.html"

if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    echo Starting Chrome in App Mode...
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --app="%HTML_PATH%" --start-maximized
    exit /b 0
)
if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    echo Starting Chrome in App Mode...
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --app="%HTML_PATH%" --start-maximized
    exit /b 0
)
if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" (
    echo Starting Chrome in App Mode...
    start "" "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" --app="%HTML_PATH%" --start-maximized
    exit /b 0
)
if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    echo Starting Edge in App Mode...
    start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --app="%HTML_PATH%" --start-maximized
    exit /b 0
)
if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    echo Starting Edge in App Mode...
    start "" "C:\Program Files\Microsoft\Edge\Application\msedge.exe" --app="%HTML_PATH%" --start-maximized
    exit /b 0
)

start "" "%HTML_PATH%"
exit /b 0
