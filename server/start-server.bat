@echo off
setlocal

cd /d "%~dp0"

if not exist "package.json" (
  echo [ERROR] package.json not found in %CD%
  pause
  exit /b 1
)

echo Starting server in %CD%

if not exist "node_modules" (
  echo Installing dependencies...
  npm install
)

npm run start
