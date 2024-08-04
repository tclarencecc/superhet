# -*- mode: python ; coding: utf-8 -*-
import shutil
import tarfile
import os

name = "main"

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[
        (".venv/lib/python3.12/site-packages/llama_cpp/lib/libllama.dylib", "llama_cpp/lib"),
        (".venv/lib/python3.12/site-packages/llama_cpp/lib/libggml.dylib", "llama_cpp/lib")
    ],
    datas=[
        ("bin/qdrant", "."),
        ("config.yaml", ".")
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=name,
)

shutil.move(f"./dist/{name}/_internal/config.yaml", f"./dist/{name}/config.yaml")

# print("----- INFO: Building tar.gz archive")
# src = f"./dist/{name}"
# with tarfile.open(f"{src}.tar.gz", "w:gz") as tar:
#    tar.add(src, arcname=os.path.basename(src))
