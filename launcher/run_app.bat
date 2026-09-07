@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python virtual environment not found at: %PYTHON_EXE%
    pause
    exit /b 1
)

echo Starting Classroom Hybrid ASR Desktop Application...
"%PYTHON_EXE%" "%~dp0..\app\main.py"

if errorlevel 1 (
    echo.
    echo Application exited with error code %ERRORLEVEL%
    pause
)
endlocal
