@echo off
setlocal

cd /d "%~dp0"

if not exist "%~dp0python_ml\start-ml-rf.bat" (
  echo [ERROR] Missing: %~dp0python_ml\start-ml-rf.bat
  pause
  exit /b 1
)

if not exist "%~dp0server\package.json" (
  echo [ERROR] Missing: %~dp0server\package.json
  pause
  exit /b 1
)

if not exist "%~dp0client\index_v2.html" (
  echo [ERROR] Missing: %~dp0client\index_v2.html
  pause
  exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
  echo [ERROR] npm was not found in PATH.
  pause
  exit /b 1
)

if not exist "%~dp0server\node_modules" (
  echo Installing server dependencies...
  pushd "%~dp0server"
  call npm install
  popd
)

echo ==========================================
echo Starting ML Inference Service (new external terminal)
echo ==========================================
start "ML Service" /D "%~dp0python_ml" cmd /k "call start-ml-rf.bat"

timeout /t 2 /nobreak >nul

echo ==========================================
echo Starting DEPLOY Server (new external terminal)
echo ==========================================
start "Deploy Server" /D "%~dp0server" cmd /k "npm run start:deploy"

timeout /t 1 /nobreak >nul

echo ==========================================
echo Opening Client
echo ==========================================
set "CLIENT_FILE=%~dp0client\index_v2.html"
set "CLIENT_URI=file:///%CLIENT_FILE:\=/%"

if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" (
  start "Client" "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" "%CLIENT_URI%"
) else if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" (
  start "Client" "%ProgramFiles%\Google\Chrome\Application\chrome.exe" "%CLIENT_URI%"
) else (
  start "" "%CLIENT_FILE%"
  if errorlevel 1 explorer "%CLIENT_FILE%"
)

echo Done.
echo If page opens before services are ready, refresh once.
