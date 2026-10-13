# Dot-source this file in the PowerShell session used to build NInfer.
param(
    [string]$CudaRoot = (Join-Path $PSScriptRoot '..\..\build-tools\cuda-12.8.1'),
    [string]$MsvcVersion = '14.44'
)

$ErrorActionPreference = 'Stop'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installations = & $vswhere -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vsRoot = $installations | Where-Object {
    @(Get-ChildItem (Join-Path $_ 'VC\Tools\MSVC') -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name.StartsWith("$MsvcVersion.") }).Count -gt 0
} | Select-Object -First 1
if (-not $vsRoot) { throw "MSVC $MsvcVersion x64 tools were not found." }
$CudaRoot = (Resolve-Path -LiteralPath $CudaRoot).Path
if (-not (Test-Path (Join-Path $CudaRoot 'bin\nvcc.exe'))) {
    throw "CUDA compiler was not found in $CudaRoot."
}

$devcmd = Join-Path $vsRoot 'Common7\Tools\VsDevCmd.bat'
$environment = & $env:ComSpec /d /s /c "`"$devcmd`" -no_logo -arch=x64 -host_arch=x64 -vcvars_ver=$MsvcVersion >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'Visual Studio environment initialization failed.' }
foreach ($line in $environment) {
    $separator = $line.IndexOf('=')
    if ($separator -gt 0) {
        [Environment]::SetEnvironmentVariable($line.Substring(0, $separator), $line.Substring($separator + 1), 'Process')
    }
}
if (-not $env:VCToolsVersion.StartsWith("$MsvcVersion.")) {
    throw "Requested MSVC $MsvcVersion, but selected $env:VCToolsVersion."
}
$cmakeRoot = Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake'
$env:CUDA_PATH = $CudaRoot
$env:CUDACXX = Join-Path $CudaRoot 'bin\nvcc.exe'
$env:CUDAHOSTCXX = (Get-Command cl.exe -ErrorAction Stop).Source
$env:CUDAToolkit_ROOT = $CudaRoot
$env:VCPKG_ROOT = Join-Path $vsRoot 'VC\vcpkg'
$env:VCPKG_MAX_CONCURRENCY = '4'
$env:CMAKE_BUILD_PARALLEL_LEVEL = '4'
# Prefer English when installed; the CMake probe also handles a Chinese-only toolset.
$env:VSLANG = '1033'
$env:PATH = "$CudaRoot\bin;$cmakeRoot\CMake\bin;$cmakeRoot\Ninja;$env:VCPKG_ROOT;$env:PATH"
Write-Host "CUDA: $CudaRoot"
Write-Host "MSVC: $env:VCToolsVersion"
Write-Host 'Environment changes apply only to this PowerShell process and its children.'
