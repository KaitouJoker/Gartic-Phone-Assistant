@echo off
setlocal
title Gartic Phone Virtual Canvas (App Mode)
cd /d "%~dp0"

echo ======================================================================
echo  Gartic Phone Virtual Canvas [App Mode]
echo  - Runs in standalone window without browser address bar or tabs.
echo  - Isolated Chrome Profile: External app links will NOT overwrite!
echo ======================================================================

set "HTML_PATH=%~dp0gartic_phone_canvas.html"
set "PROFILE_DIR=%TEMP%\gartic_canvas_profile"

if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    echo Starting Chrome in App Mode (Isolated Profile)...
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --user-data-dir="%PROFILE_DIR%" --no-first-run --no-default-browser-check --app="%HTML_PATH%" --start-maximized
    exit /b 0
)
if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    echo Starting Chrome in App Mode (Isolated Profile)...
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --user-data-dir="%PROFILE_DIR%" --no-first-run --no-default-browser-check --app="%HTML_PATH%" --start-maximized
    exit /b 0
)
if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" (
    echo Starting Chrome in App Mode (Isolated Profile)...
    start "" "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" --user-data-dir="%PROFILE_DIR%" --no-first-run --no-default-browser-check --app="%HTML_PATH%" --start-maximized
    exit /b 0
)
if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    echo Starting Edge in App Mode (Isolated Profile)...
    start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --user-data-dir="%PROFILE_DIR%" --no-first-run --no-default-browser-check --app="%HTML_PATH%" --start-maximized
    exit /b 0
)
if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    echo Starting Edge in App Mode (Isolated Profile)...
    start "" "C:\Program Files\Microsoft\Edge\Application\msedge.exe" --user-data-dir="%PROFILE_DIR%" --no-first-run --no-default-browser-check --app="%HTML_PATH%" --start-maximized
    exit /b 0
)

start "" "%HTML_PATH%"
exit /b 0
