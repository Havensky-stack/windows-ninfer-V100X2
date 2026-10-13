# Compile and run the existing TP2 probe using native sm_70 machine code.
# This checks compiler/runtime/driver compatibility, not full-model inference.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-toolchain.ps1')
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$probeDir = Join-Path $repoRoot 'build-toolchain-check'
New-Item -ItemType Directory -Force $probeDir | Out-Null
Push-Location $probeDir
try {
    & nvcc --version
    if ($LASTEXITCODE -ne 0) { throw 'nvcc version check failed.' }
    & cmake --version
    & ninja --version
    & nvidia-smi --query-gpu=index,name,driver_version,driver_model.current,pcie.link.width.current --format=csv
    & cmake -S (Join-Path $PSScriptRoot 'probe') -B $probeDir -G Ninja `
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70-real `
        "-DCMAKE_CUDA_COMPILER=$env:CUDACXX" "-DCMAKE_CUDA_HOST_COMPILER=$env:CUDAHOSTCXX" `
        "-DCUDAToolkit_ROOT=$env:CUDA_PATH"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA TP2 probe configuration failed.' }
    & cmake --build $probeDir --parallel 2
    if ($LASTEXITCODE -ne 0) { throw 'CUDA TP2 probe compilation failed.' }
    & .\capture_probe.exe 0 1
    if ($LASTEXITCODE -ne 0) { throw "CUDA TP2 probe failed with exit code $LASTEXITCODE." }
} finally {
    Pop-Location
}
