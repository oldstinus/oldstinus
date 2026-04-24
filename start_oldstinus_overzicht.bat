@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON_EXE=%ROOT%.venv_oldstinus\Scripts\python.exe"
set "CATALOG_SCRIPT=%ROOT%generate_oldstinus_catalog.py"
set "OUTPUT_HTML=%ROOT%oldstinus_overzicht.html"

if exist "%PYTHON_EXE%" (
  "%PYTHON_EXE%" "%CATALOG_SCRIPT%"
) else (
  python "%CATALOG_SCRIPT%"
)

start "" "%OUTPUT_HTML%"
