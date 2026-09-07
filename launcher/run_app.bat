@echo off
chcp 65001 >nul
title 课堂实时转写 (Hybrid 2-Pass ASR)

cd /d "%~dp0\.."

set "PYTHON_EXE=%~dp0\..\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [错误] 虚拟环境不存在: %PYTHON_EXE%
    pause
    exit /b 1
)

echo =================================================================
echo        正在启动 课堂实时转写系统 (Hybrid 2-Pass ASR)
echo =================================================================
"%PYTHON_EXE%" "app\main.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [异常退出] 程序退出代码: %ERRORLEVEL%
    pause
)
