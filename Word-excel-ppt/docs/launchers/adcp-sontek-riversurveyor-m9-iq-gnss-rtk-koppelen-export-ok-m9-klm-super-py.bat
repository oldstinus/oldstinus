@echo off
setlocal
set "PROJECT_DIR=C:\Users\claeysst\AppData\Roaming\Python\Python312\Scripts\oldstinus\ADCP_Sontek_Riversurveyor_M9_IQ\GNSS_RTK_koppelen_export"
set "SCRIPT_PATH=C:\Users\claeysst\AppData\Roaming\Python\Python312\Scripts\oldstinus\ADCP_Sontek_Riversurveyor_M9_IQ\GNSS_RTK_koppelen_export\OK_m9-klm_SUPER.py"
set "CODE_EXE=C:\Users\claeysst\AppData\Local\Programs\Microsoft VS Code\Code.exe"
set "VENV_ACTIVATE=C:\Users\claeysst\AppData\Roaming\Python\Python312\Scripts\oldstinus\.venv\Scripts\activate.bat"

if exist "%CODE_EXE%" (
  start "" "%CODE_EXE%" --new-window "%PROJECT_DIR%" "%SCRIPT_PATH%"
) else (
  echo VS Code niet gevonden op "%CODE_EXE%".
)

if exist "%VENV_ACTIVATE%" (
  start "oldstinus-script" cmd /K "cd /d ""%PROJECT_DIR%"" && call ""%VENV_ACTIVATE%"" && python ""%SCRIPT_PATH%"""
) else (
  start "oldstinus-script" cmd /K "cd /d ""%PROJECT_DIR%"" && python ""%SCRIPT_PATH%"""
)
