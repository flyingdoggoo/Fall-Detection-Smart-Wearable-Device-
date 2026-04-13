@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0start-all-deploy.bat" (
	echo [ERROR] Missing: %~dp0start-all-deploy.bat
	pause
	exit /b 1
)

call "%~dp0start-all-deploy.bat"
