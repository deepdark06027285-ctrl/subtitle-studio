param([string]$DestinationDirectory)
$ErrorActionPreference = 'Stop'
$subtitleProject = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$subtitleConfig = Get-Content -LiteralPath (Join-Path $subtitleProject 'config.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$subtitlePython = Join-Path (Split-Path -Parent ([string]$subtitleConfig.python)) 'pythonw.exe'
$subtitleApp = Join-Path $subtitleProject 'app.py'
$subtitleIcon = Join-Path $subtitleProject 'assets\subtitle-studio.ico'
foreach ($subtitleFile in @($subtitlePython,$subtitleApp,$subtitleIcon)) {
    if (-not (Test-Path -LiteralPath $subtitleFile -PathType Leaf)) { throw "缺少启动文件：$subtitleFile" }
}
if (-not $DestinationDirectory) { $DestinationDirectory = [Environment]::GetFolderPath('Desktop') }
$subtitleDestination = (Resolve-Path -LiteralPath $DestinationDirectory).Path
$subtitleShell = New-Object -ComObject WScript.Shell
$subtitleShortcutPath = Join-Path $subtitleDestination '字幕工坊.lnk'
$subtitleSuffix = 1
while (Test-Path -LiteralPath $subtitleShortcutPath) {
    $subtitleExisting = $subtitleShell.CreateShortcut($subtitleShortcutPath)
    if ($subtitleExisting.TargetPath -eq $subtitlePython -and $subtitleExisting.Arguments -eq ('"' + $subtitleApp + '"')) { break }
    $subtitleShortcutPath = Join-Path $subtitleDestination "字幕工坊 ($subtitleSuffix).lnk"
    $subtitleSuffix++
}
$subtitleShortcut = $subtitleShell.CreateShortcut($subtitleShortcutPath)
$subtitleShortcut.TargetPath = $subtitlePython
$subtitleShortcut.Arguments = '"' + $subtitleApp + '"'
$subtitleShortcut.WorkingDirectory = $subtitleProject
$subtitleShortcut.IconLocation = "$subtitleIcon,0"
$subtitleShortcut.Description = '本地日语视频转中文字幕 · 自动显卡检测 / CPU+GPU 协同'
$subtitleShortcut.WindowStyle = 1
$subtitleShortcut.Save()
$subtitleCheck = $subtitleShell.CreateShortcut($subtitleShortcutPath)
if ($subtitleCheck.TargetPath -ne $subtitlePython -or $subtitleCheck.Arguments -ne ('"' + $subtitleApp + '"') -or $subtitleCheck.WorkingDirectory -ne $subtitleProject) { throw '快捷方式验证失败。' }
[pscustomobject]@{ Shortcut=$subtitleShortcutPath; Target=$subtitleCheck.TargetPath; Arguments=$subtitleCheck.Arguments; WorkingDirectory=$subtitleCheck.WorkingDirectory; Icon=$subtitleCheck.IconLocation } | ConvertTo-Json -Compress
