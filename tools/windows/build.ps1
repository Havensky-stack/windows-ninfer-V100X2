param(
    [ValidateRange(1, 16)][int]$BuildJobs = 4,
    [switch]$Tests,
    [string]$PythonExecutable
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-toolchain.ps1')
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$buildDir = Join-Path $repoRoot 'build-windows'
$env:VCPKG_MAX_CONCURRENCY = "$BuildJobs"
$configure = @('-S', $repoRoot, '-B', $buildDir, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=70',
    "-DCMAKE_CUDA_COMPILER=$env:CUDACXX", "-DCMAKE_CUDA_HOST_COMPILER=$env:CUDAHOSTCXX",
    "-DCUDAToolkit_ROOT=$env:CUDA_PATH",
    "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake",
    '-DVCPKG_TARGET_TRIPLET=x64-windows-ninfer',
    "-DVCPKG_OVERLAY_TRIPLETS=$PSScriptRoot\triplets",
    "-DVCPKG_INSTALLED_DIR=$buildDir\vcpkg_installed",
    '-DNINFER_BUILD_APPS=ON', "-DBUILD_TESTING=$($Tests.IsPresent.ToString().ToUpperInvariant())")
if ($PythonExecutable) { $configure += "-DPython3_EXECUTABLE=$PythonExecutable" }
$cutlass = Join-Path $repoRoot 'build-tools\cutlass'
if (Test-Path (Join-Path $cutlass '.git')) {
    $revision = & git -C $cutlass rev-parse HEAD
    if ($LASTEXITCODE -eq 0 -and $revision -eq 'da5e086dab31d63815acafdac9a9c5893b1c69e2') {
        $configure += "-DFETCHCONTENT_SOURCE_DIR_CUTLASS=$cutlass"
    }
}
& cmake @configure
if ($LASTEXITCODE -ne 0) { throw 'NInfer configuration failed.' }
# Build product binaries only. Select focused test targets explicitly after configuration.
& cmake --build $buildDir --parallel $BuildJobs --target ninfer ninfer-serve
if ($LASTEXITCODE -ne 0) { throw 'NInfer build failed.' }
Write-Host "CLI: $buildDir\apps\ninfer.exe"
Write-Host "API: $buildDir\apps\ninfer-serve.exe"
