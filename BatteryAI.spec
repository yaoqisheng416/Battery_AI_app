# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all
from glob import glob
import os
import torch
import sys


block_cipher = None

# =========================================
# �Զ��ռ� torch / fastapi ������
# =========================================

torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')

fastapi_datas, fastapi_binaries, fastapi_hiddenimports = collect_all('fastapi')

uvicorn_datas, uvicorn_binaries, uvicorn_hiddenimports = collect_all('uvicorn')

lightning_datas, lightning_binaries, lightning_hiddenimports = collect_all('lightning')

scipy_datas, scipy_binaries, scipy_hiddenimports = collect_all('scipy')

skimage_datas, skimage_binaries, skimage_hiddenimports = collect_all('skimage')


a = Analysis(
    ['OpenMesoCell V0.1.py'],

    pathex=[],

    binaries=[
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
        # 项目根目录下被动态导入的模块（pages/backend 通过 __import__ 或
        # from xxx import 引用，PyInstaller 静态分析追踪不到，需显式声明）
        ('api_client.py', '.'),
        ('config.py', '.'),
        ('latent_diffusion.py', '.'),
        ('vaemodule.py', '.'),
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

    name='OpenMesoCell V0.9',

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

    name='OpenMesoCell V0.9'
)