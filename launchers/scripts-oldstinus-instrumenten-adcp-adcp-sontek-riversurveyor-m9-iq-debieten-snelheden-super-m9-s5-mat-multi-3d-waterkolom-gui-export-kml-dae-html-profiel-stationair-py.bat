@echo off
setlocal
set "PROJECT_DIR=C:\Users\claeysst\AppData\Roaming\Python\Python312\Scripts\oldstinus\Instrumenten\ADCP\ADCP_Sontek_Riversurveyor_M9_IQ\Debieten&snelheden"
set "SCRIPT_PATH=C:\Users\claeysst\AppData\Roaming\Python\Python312\Scripts\oldstinus\Instrumenten\ADCP\ADCP_Sontek_Riversurveyor_M9_IQ\Debieten&snelheden\SUPER_M9_S5_MAT_multi_3D_waterkolom_gui_export_kml_dae_HTML_profiel_stationair.py"
set "CODE_EXE=C:\Users\claeysst\AppData\Local\Programs\Microsoft VS Code\Code.exe"
set "VENV_ACTIVATE=C:\Users\claeysst\AppData\Roaming\Python\Python312\Scripts\oldstinus\Instrumenten\ADCP\ADCP_Sontek_Riversurveyor_M9_IQ\.venv_adcp_sontek_m9_iq\Scripts\activate.bat"

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
