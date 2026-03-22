@echo off
setlocal

cd /d "%~dp0"

set "PYTHON312=C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe"
set "PYTHON_CMD="

if exist "%PYTHON312%" (
  set "PYTHON_CMD=%PYTHON312%"
)

if not defined PYTHON_CMD (
  where python >nul 2>&1
  if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
  where py >nul 2>&1
  if not errorlevel 1 set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD (
  echo [ERROR] Python 3 was not found in PATH.
  pause
  exit /b 1
)

echo Starting ML service in %CD%
echo Using interpreter: %PYTHON_CMD%
%PYTHON_CMD% app.py
