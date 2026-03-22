@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_CMD="

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

echo Starting GPU ML service in %CD%
echo Using interpreter: %PYTHON_CMD%
echo Note: TensorFlow GPU works on Linux or WSL2 for TF 2.11+.
%PYTHON_CMD% app_gpu.py
