# coding: utf-8
"""Build a native desktop bundle on the current platform.

PyInstaller 不能用一台机器生成三个平台的二进制；请在 Windows/macOS/Linux 各自
执行一次本脚本。模型权重仍在运行时从 ModelScope/用户配置目录加载，不塞进 exe。
"""
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    try:
        import PyInstaller.__main__
    except ImportError as exc:
        raise SystemExit("请先安装构建依赖：python -m pip install pyinstaller") from exc

    root = Path(__file__).resolve().parent
    dist_name = "ClassroomASR"
    args = [
        str(root / "main.py"),
        "--noconfirm",
        "--clean",
    ]
    if sys.platform == "darwin":
        # macOS 交付 .app/.dmg；Windows/Linux 交付单文件可执行程序。
        args.extend(["--windowed"])
    else:
        args.extend(["--windowed", "--onefile"])
    args.extend([
        "--name",
        dist_name,
        "--paths",
        str(root),
        "--add-data",
        f"{root / 'courses'}{os.pathsep}courses",
        "--add-data",
        f"{root / 'config' / 'settings.json'}{os.pathsep}config",
    ])
    for package in ("funasr", "modelscope", "transformers"):
        args.extend(["--collect-all", package])
    # qwen-asr 是可选的本地第二遍包；没有安装时仍允许构建 Paraformer +
    # CapsWriter 回退版，而装好 qwen-asr 时会把它完整收进包内。
    try:
        __import__("qwen_asr")
    except ImportError:
        print("qwen-asr 未安装：将构建带 CapsWriter/Paraformer 回退的版本")
    else:
        args.extend(["--collect-all", "qwen_asr"])
    PyInstaller.__main__.run(args)
    output = root / "dist" / dist_name
    if sys.platform == "darwin":
        app_path = root / "dist" / f"{dist_name}.app"
        if not app_path.exists():
            # 某些 PyInstaller 版本会生成同名的 onedir 目录。
            app_path = root / "dist" / dist_name
        dmg_path = root / "dist" / f"{dist_name}.dmg"
        if not app_path.exists():
            raise SystemExit(f"PyInstaller 未生成 macOS app bundle: {app_path}")
        if dmg_path.exists():
            dmg_path.unlink()
        subprocess.run(
            [
                "hdiutil", "create", "-volname", dist_name,
                "-srcfolder", str(app_path), "-ov", "-format", "UDZO",
                str(dmg_path),
            ],
            check=True,
        )
        print(f"Build complete: {dmg_path}")
    else:
        print(f"Build complete: {output}.exe" if sys.platform == "win32" else f"Build complete: {output}")


if __name__ == "__main__":
    main()
