@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title Gartic Phone 가상 캔버스 (키오스크 전체화면 모드)

echo ======================================================================
echo  Gartic Phone 가상 캔버스 [키오스크 모드] 실행기
echo.
echo  * 특징: 브라우저 상단 X버튼 / 팝업이 100%% 차단된 완전한 전체화면입니다.
echo  * 캔버스 영역: 모니터 0px부터 화면 끝까지 100%% 캔버스로 채워집니다.
echo  * 종료 방법: 키보드의 [Alt + F4] 키를 누르면 종료됩니다.
echo ======================================================================

set "HTML_PATH=%~dp0gartic_phone_canvas.html"

:: 1. Google Chrome 탐색
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    echo Chrome 키오스크 모드로 실행합니다...
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --kiosk "file:///%HTML_PATH%"
    exit /b 0
)
if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    echo Chrome 키오스크 모드로 실행합니다...
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --kiosk "file:///%HTML_PATH%"
    exit /b 0
)
if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" (
    echo Chrome 키오스크 모드로 실행합니다...
    start "" "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" --kiosk "file:///%HTML_PATH%"
    exit /b 0
)

:: 2. Microsoft Edge 탐색
if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    echo Edge 키오스크 모드로 실행합니다...
    start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --kiosk "file:///%HTML_PATH%"
    exit /b 0
)
if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    echo Edge 키오스크 모드로 실행합니다...
    start "" "C:\Program Files\Microsoft\Edge\Application\msedge.exe" --kiosk "file:///%HTML_PATH%"
    exit /b 0
)

:: 3. Brave Browser 탐색
if exist "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" (
    echo Brave 키오스크 모드로 실행합니다...
    start "" "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" --kiosk "file:///%HTML_PATH%"
    exit /b 0
)

:: 4. 기본 브라우저 실행
echo 브라우저 경로를 직접 찾지 못해 기본 브라우저로 엽니다...
start "" "%HTML_PATH%"
exit /b 0
