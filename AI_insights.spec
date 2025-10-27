# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=[('model\\clustering_model.joblib', 'model'), ('model\\scaler.joblib', 'model'), ('data\\training_report.md', 'data')],
    hiddenimports=[
        'src',
        'src.clustering_engine',
        'src.data_processor_clustering',
        'src.predictor_clustering',
        'src.services.file_service',
        'src.services.insight_service',
        'src.services.report_service',
    ],
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
    a.binaries,
    a.datas,
    [],
    name='AI_insights',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['ai_ico.ico'],
)
