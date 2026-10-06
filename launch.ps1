$ErrorActionPreference = 'Stop'
try {
    $settings = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'config.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $pythonPath = [string]$settings.python
    if (-not (Test-Path -LiteralPath $pythonPath)) { throw '找不到 Python，请先运行安装环境.cmd。' }
    $windowedPython = Join-Path (Split-Path -Parent $pythonPath) 'pythonw.exe'
    if (-not (Test-Path -LiteralPath $windowedPython)) { $windowedPython = $pythonPath }
    $appPath = Join-Path $PSScriptRoot 'app.py'
    Start-Process -FilePath $windowedPython -ArgumentList @('"' + $appPath + '"') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
} catch {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, '本地字幕启动失败') | Out-Null
}
