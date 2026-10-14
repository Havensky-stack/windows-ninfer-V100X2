param([string]$Version = '0.1.3', [switch]$SkipArchive)
$ErrorActionPreference = 'Stop'
if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw 'Use a numeric major.minor.patch version.' }
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$releaseRoot = Join-Path $repo 'build-release'
$package = Join-Path $releaseRoot "NInfer-Windows-V100X2-$Version"
$bin = Join-Path $package 'bin'
$licenses = Join-Path $package 'licenses'
New-Item -ItemType Directory -Force $bin, $licenses | Out-Null
$apps = Join-Path $repo 'build-windows\apps'
# The GUI only needs the server; avoid shipping a second copy of the CUDA kernels.
foreach ($name in @('ninfer-serve.exe', 'avcodec-62.dll', 'avformat-62.dll', 'avutil-60.dll', 'swscale-9.dll', 'libcurl.dll', 'z.dll')) {
    Copy-Item -LiteralPath (Join-Path $apps $name) -Destination $bin -Force
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products * -property installationPath
$crt = Join-Path $vs 'VC\Redist\MSVC\14.44.35112\x64\Microsoft.VC143.CRT'
if (-not (Test-Path $crt)) { throw 'Required MSVC 14.44.35112 redistributable directory not found.' }
Copy-Item -Path (Join-Path $crt '*.dll') -Destination $bin -Force
$csc = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
& $csc /nologo /target:winexe /platform:x64 /optimize+ /utf8output `
    "/out:$package\NInfer.exe" "/win32manifest:$PSScriptRoot\launcher\app.manifest" `
    /reference:System.Windows.Forms.dll /reference:System.Drawing.dll /reference:System.Net.Http.dll `
    /reference:System.Web.Extensions.dll (Join-Path $PSScriptRoot 'launcher\Launcher.cs')
if ($LASTEXITCODE -ne 0) { throw 'Launcher compilation failed.' }
Copy-Item -LiteralPath (Join-Path $repo 'LICENSE'), (Join-Path $repo 'NOTICE') -Destination $package -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'launcher\使用说明.txt') -Destination $package -Force
$modeTools = Join-Path $package 'tools'
New-Item -ItemType Directory -Force $modeTools | Out-Null
foreach ($name in @('enable-tcc.cmd', 'restore-wddm.cmd', 'set-v100-mode.ps1')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination $modeTools -Force
}
Copy-Item -LiteralPath (Join-Path $repo 'build-tools\cuda-12.8.1\LICENSE') -Destination (Join-Path $licenses 'NVIDIA-CUDA.txt') -Force
foreach ($dependency in @('ffmpeg', 'curl', 'zlib')) {
    $share = Join-Path $repo "build-windows\vcpkg_installed\x64-windows-ninfer\share\$dependency"
    Copy-Item -LiteralPath (Join-Path $share 'copyright') -Destination (Join-Path $licenses "$dependency.txt") -Force
    Copy-Item -LiteralPath (Join-Path $share 'vcpkg.spdx.json') -Destination (Join-Path $licenses "$dependency.spdx.json") -Force
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'launcher\THIRD-PARTY.md') -Destination $licenses -Force
foreach ($entry in @(
    @('third_party\utf8proc\LICENSE.md', 'utf8proc.txt'),
    @('third_party\nlohmann\LICENSE.MIT', 'nlohmann-json.txt'),
    @('third_party\cpp-httplib\LICENSE', 'cpp-httplib.txt'),
    @('third_party\llama_cpp_fattn\LICENSE', 'llama-cpp-fattn.txt'),
    @('build-tools\cutlass\LICENSE.txt', 'CUTLASS.txt'))) {
    Copy-Item -LiteralPath (Join-Path $repo $entry[0]) -Destination (Join-Path $licenses $entry[1]) -Force
}
Get-Content (Join-Path $repo 'build-tools\cuda-12.8.1\include\nvtx3\nvToolsExt.h') -TotalCount 38 |
    Set-Content (Join-Path $licenses 'NVTX-notice.txt') -Encoding utf8
if (-not $SkipArchive) {
    $archive = Join-Path $releaseRoot "NInfer-Windows-V100X2-$Version.zip"
    Compress-Archive -LiteralPath $package -DestinationPath $archive -Force -CompressionLevel Optimal
    $hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $([IO.Path]::GetFileName($archive))" | Set-Content (Join-Path $releaseRoot 'SHA256SUMS.txt') -Encoding ascii
    Write-Output $archive
}
Write-Output $package
