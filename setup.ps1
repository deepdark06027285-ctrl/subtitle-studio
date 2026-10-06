$ErrorActionPreference = 'Stop'
try {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { throw '请先安装 Python 3.12（含 Tcl/Tk），再重新运行。' }
    $pythonPath = $pythonCommand.Source
    & $pythonPath -m pip install --target (Join-Path $PSScriptRoot 'vendor') -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw '运行库安装失败。' }
    & $pythonPath -m pip install --target (Join-Path $PSScriptRoot 'ocr-vendor') -r (Join-Path $PSScriptRoot 'ocr-requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw '画面日语识别环境安装失败。' }
    & $pythonPath -m pip install --target (Join-Path $PSScriptRoot 'ocr-gpu-vendor') --no-deps paddlepaddle-gpu==3.2.0 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/
    if ($LASTEXITCODE -ne 0) { Write-Output '画面显卡运行库未装好；请重新安装，或手动选择 CPU 模式。' }
    & $pythonPath -m pip install --target (Join-Path $PSScriptRoot 'ocr-gpu-cuda') nvidia-cudnn-cu11==8.9.5.29 nvidia-cublas-cu11==11.11.3.6 nvidia-cuda-runtime-cu11==11.8.89
    if ($LASTEXITCODE -ne 0) { Write-Output '画面显卡依赖未装好；请重新安装，或手动选择 CPU 模式。' }
    & $pythonPath -m pip install --target (Join-Path $PSScriptRoot 'cuda') nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12
    if ($LASTEXITCODE -ne 0) { Write-Output '显卡库安装失败，可使用 CPU 模式。' }
    & $pythonPath (Join-Path $PSScriptRoot 'setup_models.py')
    if ($LASTEXITCODE -ne 0) { throw '模型下载未完成，请检查网络后重试。' }
    Write-Output '安装完成。现在可以双击启动字幕工具.cmd。'
} catch {
    Write-Output $_.Exception.Message
    exit 1
}
