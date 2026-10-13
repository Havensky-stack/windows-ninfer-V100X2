$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$output = Join-Path $repo 'build-release\launcher-tests.exe'
New-Item -ItemType Directory -Force (Split-Path $output) | Out-Null
$csc = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
& $csc /nologo /target:exe /platform:x64 /optimize+ /utf8output /main:NInferWindows.LauncherTests "/out:$output" `
    /reference:System.Windows.Forms.dll /reference:System.Drawing.dll /reference:System.Net.Http.dll /reference:System.Web.Extensions.dll `
    (Join-Path $PSScriptRoot 'launcher\Launcher.cs') (Join-Path $repo 'tests\windows\test_launcher.cs')
if ($LASTEXITCODE -ne 0) { throw 'Launcher test compilation failed.' }
& $output
if ($LASTEXITCODE -ne 0) { throw 'Launcher download checks failed.' }
