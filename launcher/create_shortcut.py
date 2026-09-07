# coding: utf-8
"""
创建 Windows 桌面快捷方式: 课堂实时转写.lnk
"""
import os
import sys
import subprocess
from pathlib import Path

def create_desktop_shortcut():
    desktop = Path(os.environ.get("USERPROFILE", "C:\\Users\\david")) / "Desktop"
    if not desktop.exists():
        # Fallback to OneDrive Desktop if synced
        onedrive_desktop = Path(os.environ.get("USERPROFILE", "C:\\Users\\david")) / "OneDrive" / "Desktop"
        if onedrive_desktop.exists():
            desktop = onedrive_desktop

    root_dir = Path(__file__).resolve().parent.parent
    target_bat = root_dir / "launcher" / "run_app.bat"
    pythonw_exe = root_dir / ".venv" / "Scripts" / "pythonw.exe"
    main_py = root_dir / "app" / "main.py"
    shortcut_path = desktop / "课堂实时转写.lnk"

    # PowerShell script to create .lnk shortcut
    # Target: pythonw.exe, Arguments: "d:\课程笔记\app\main.py", WorkingDirectory: "d:\课程笔记"
    ps_cmd = f"""
    $WshShell = New-Object -ComObject WScript.Shell;
    $Shortcut = $WshShell.CreateShortcut('{shortcut_path}');
    $Shortcut.TargetPath = '{pythonw_exe}';
    $Shortcut.Arguments = '"{main_py}"';
    $Shortcut.WorkingDirectory = '{root_dir}';
    $Shortcut.Description = '课堂实时转写 (Hybrid 2-Pass Classroom ASR)';
    $Shortcut.Save();
    """

    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=True)
        print(f"[OK] 桌面快捷方式已成功创建: {shortcut_path}")
        return True
    except Exception as e:
        print(f"[FAIL] 创建桌面快捷方式失败: {e}")
        return False

if __name__ == "__main__":
    create_desktop_shortcut()
