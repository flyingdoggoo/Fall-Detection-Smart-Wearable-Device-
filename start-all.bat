@echo off
setlocal

REM Start everything from repo root
cd /d "%~dp0"

if not exist "%~dp0server\start-server.bat" (
	echo [ERROR] Missing: %~dp0server\start-server.bat
	echo Recreate it or check your repo structure.
	pause
	exit /b 1
)

if not exist "%~dp0client\index_v2.html" (
	echo [ERROR] Missing: %~dp0client\index_v2.html
	pause
	exit /b 1
)

if not exist "%~dp0python_ml\start-ml.bat" (
	echo [ERROR] Missing: %~dp0python_ml\start-ml.bat
	pause
	exit /b 1
)

echo ==========================================
echo Starting ML Inference Service (Python)
echo ==========================================

start "ML Service" cmd /k ""%~dp0python_ml\start-ml.bat""

timeout /t 2 /nobreak >nul

echo ==========================================
echo Starting Sensor Server (Node.js)
echo ==========================================

REM Open server in its own window
start "Sensor Server" cmd /k ""%~dp0server\start-server.bat""

echo ==========================================
echo Opening Sensor Client (HTML)
echo ==========================================

REM Open the HTML client in default browser
start "" "%~dp0client\index_v2.html"

echo Done.
echo (If the browser opened before the server is ready, just refresh the page.)
