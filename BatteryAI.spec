# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all
from glob import glob
import os
import torch
import sys

# ===== 添加这部分 =====
python_dll_path = os.path.join(sys.prefix, 'python311.dll')
python_dlls = [(python_dll_path, '.')] if os.path.exists(python_dll_path) else []

block_cipher = None
torch_lib_path = os.path.join(
    os.path.dirname(torch.__file__),
    'lib'
)

torch_dlls = [
    (dll, '.')
    for dll in glob(os.path.join(torch_lib_path, '*.dll'))
]

# =========================================
# 自动收集 torch / fastapi 等依赖
# =========================================

torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')

fastapi_datas, fastapi_binaries, fastapi_hiddenimports = collect_all('fastapi')

uvicorn_datas, uvicorn_binaries, uvicorn_hiddenimports = collect_all('uvicorn')

lightning_datas, lightning_binaries, lightning_hiddenimports = collect_all('lightning')

scipy_datas, scipy_binaries, scipy_hiddenimports = collect_all('scipy')

skimage_datas, skimage_binaries, skimage_hiddenimports = collect_all('skimage')


a = Analysis(
    ['main.py'],

    pathex=[],

binaries=[
    *python_dlls,
    *torch_dlls,

    *torch_binaries,
    *fastapi_binaries,
    *uvicorn_binaries,
    *lightning_binaries,
    *scipy_binaries,
    *skimage_binaries,
],

    datas=[
        ('backend', 'backend'),
        ('pages', 'pages'),
        ('workspace', 'workspace'),
        ('latent_diffusion.py', '.'),
        ('vaemodule.py', '.'),
        ('backend/electrode_twin/checkpoints', 'checkpoints'),
        ('backend/electrode_twin/ldm_checkpoints', 'ldm_checkpoints'),
        *torch_datas,
        *fastapi_datas,
        *uvicorn_datas,
        *lightning_datas,
        *scipy_datas,
        *skimage_datas,
    ],

    hiddenimports=[
        *torch_hiddenimports,
        *fastapi_hiddenimports,
        *uvicorn_hiddenimports,
        *lightning_hiddenimports,
        *scipy_hiddenimports,
        *skimage_hiddenimports,

        'backend',
        'backend.api',
        'backend.electrode_twin',
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.websockets',
        'uvicorn.lifespan',
        'fastapi',
        'pydantic',
        'anyio',
        'starlette',
    ],

    hookspath=[],

    hooksconfig={},

    runtime_hooks=[],

    excludes=[],

    win_no_prefer_redirects=False,

    win_private_assemblies=False,

    cipher=block_cipher,

    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,

    a.scripts,

    [],

    exclude_binaries=True,

    name='BatteryAI',

    debug=False,

    bootloader_ignore_signals=False,

    strip=False,

    upx=True,

    console=False,
)

coll = COLLECT(
    exe,

    a.binaries,

    a.zipfiles,

    a.datas,

    strip=False,

    upx=True,

    upx_exclude=[],

    name='BatteryAI'
)