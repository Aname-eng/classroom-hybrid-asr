@echo off
chcp 65001 >nul
title 课堂实时转写 (Hybrid 2-Pass ASR)

cd /d "%~dp0\.."

set "PYTHON_EXE=%~dp0\..\.venv\Scripts\python.exe"
set "PYTHONW_EXE=%~dp0\..\.venv\Scripts\pythonw.exe"

if not exist "%PYTHON_EXE%" (
    echo [错误] 虚拟环境不存在: %PYTHON_EXE%
    pause
    exit /b 1
)

echo [启动] 正在启动课堂实时转写应用...
start "" "%PYTHONW_EXE%" "app\main.py"
exit /b 0
