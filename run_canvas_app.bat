@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title Gartic Phone 가상 캔버스 (앱 모드)

echo ======================================================================
echo  Gartic Phone 가상 캔버스 [앱 모드] 실행기
echo  - 주소창과 탭이 없는 독립 데스크톱 창 형태로 실행됩니다.
echo ======================================================================

set "HTML_PATH=%~dp0gartic_phone_canvas.html"

if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --app="file:///%HTML_PATH%" --start-maximized
    exit /b 0
)
if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --app="file:///%HTML_PATH%" --start-maximized
    exit /b 0
)
if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" (
    start "" "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" --app="file:///%HTML_PATH%" --start-maximized
    exit /b 0
)
if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --app="file:///%HTML_PATH%" --start-maximized
    exit /b 0
)
if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    start "" "C:\Program Files\Microsoft\Edge\Application\msedge.exe" --app="file:///%HTML_PATH%" --start-maximized
    exit /b 0
)

start "" "%HTML_PATH%"
exit /b 0
