@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"

if exist "%PYTHON_EXE%" (
  "%PYTHON_EXE%" "%ROOT%docs\generate_oldstinus_catalog.py"
) else (
  python "%ROOT%docs\generate_oldstinus_catalog.py"
)

start "" "%ROOT%docs\oldstinus_overzicht.html"
