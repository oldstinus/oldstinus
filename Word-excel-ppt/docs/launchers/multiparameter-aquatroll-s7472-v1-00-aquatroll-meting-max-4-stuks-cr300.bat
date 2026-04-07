@echo off
setlocal
set "PROJECT_DIR=C:\Users\claeysst\AppData\Roaming\Python\Python312\Scripts\oldstinus\Multiparameter_Aquatroll"
set "SCRIPT_PATH=C:\Users\claeysst\AppData\Roaming\Python\Python312\Scripts\oldstinus\Multiparameter_Aquatroll\S7472 V1_00 Aquatroll meting max 4 stuks.CR300"
set "CODE_EXE=C:\Users\claeysst\AppData\Local\Programs\Microsoft VS Code\Code.exe"
set "VENV_ACTIVATE=C:\Users\claeysst\AppData\Roaming\Python\Python312\Scripts\oldstinus\.venv\Scripts\activate.bat"

if exist "%CODE_EXE%" (
  start "" "%CODE_EXE%" --new-window "%PROJECT_DIR%" "%SCRIPT_PATH%"
) else (
  echo VS Code niet gevonden op "%CODE_EXE%".
)

if exist "%VENV_ACTIVATE%" (
  start "oldstinus-script" cmd /K "cd /d ""%PROJECT_DIR%"" && call ""%VENV_ACTIVATE%"" && echo Klaar in projectmap: %PROJECT_DIR% && echo Bestand: %SCRIPT_PATH%"
) else (
  start "oldstinus-script" cmd /K "cd /d ""%PROJECT_DIR%"" && echo Klaar in projectmap: %PROJECT_DIR% && echo Bestand: %SCRIPT_PATH%"
)
