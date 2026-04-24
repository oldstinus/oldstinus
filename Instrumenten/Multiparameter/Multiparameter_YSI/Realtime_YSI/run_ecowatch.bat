@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%.venv_ysi_realtime\Scripts\python.exe"
set "APP_FILE=%SCRIPT_DIR%EcoWatchClone.py"

if not exist "%PYTHON_EXE%" (
  echo Python venv niet gevonden: "%PYTHON_EXE%"
  exit /b 1
)

if not exist "%APP_FILE%" (
  echo App-bestand niet gevonden: "%APP_FILE%"
  exit /b 1
)

cd /d "%SCRIPT_DIR%"
"%PYTHON_EXE%" "%APP_FILE%"

endlocal
