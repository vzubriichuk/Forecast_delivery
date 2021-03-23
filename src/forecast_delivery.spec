# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# need additional import for sklearn module
from PyInstaller.utils.hooks import collect_submodules
hidden_imports_sklearn = collect_submodules('sklearn')
hidden_imports_sklalchemy = collect_submodules('sqlalchemy')
all_hidden_imports = hidden_imports_sklearn + hidden_imports_sklalchemy


a = Analysis(['forecast_delivery.py'],
             pathex=['C:\\Work\\Python_projects\\forecast_delivery\\src'],
             binaries=[],
             datas=[( '.\\resources\\*.ico', 'resources' )],
             hiddenimports=all_hidden_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='forecast_delivery',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True , icon='resources\\box.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='forecast_delivery')