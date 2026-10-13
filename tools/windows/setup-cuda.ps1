# Assemble NVIDIA's CUDA 12.8.1 redistributable components locally.
# Does not install a display driver or change persistent environment variables.
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$toolsRoot = Join-Path $repoRoot 'build-tools'
$downloads = Join-Path $toolsRoot 'downloads'
$unpacked = Join-Path $toolsRoot 'unpacked'
$cudaRoot = Join-Path $toolsRoot 'cuda-12.8.1'
New-Item -ItemType Directory -Force $downloads, $unpacked, $cudaRoot | Out-Null
$base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
$manifest = Invoke-RestMethod ($base + 'redistrib_12.8.1.json')
$manifest | ConvertTo-Json -Depth 12 | Set-Content (Join-Path $downloads 'redistrib_12.8.1.json')
foreach ($component in @('cuda_nvcc', 'cuda_cudart', 'cuda_cccl', 'cuda_nvml_dev', 'cuda_nvtx')) {
    $package = $manifest.$component.'windows-x86_64'
    if (-not $package.relative_path -or -not $package.sha256) {
        throw "Missing Windows package metadata: $component"
    }
    $archive = Join-Path $downloads ([IO.Path]::GetFileName($package.relative_path))
    $valid = (Test-Path $archive) -and ((Get-FileHash $archive -Algorithm SHA256).Hash -eq $package.sha256)
    if (-not $valid) { Invoke-WebRequest ($base + $package.relative_path) -OutFile $archive }
    if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne $package.sha256) {
        throw "SHA-256 mismatch: $component"
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $unpacked -Force
    $packageRoot = Join-Path $unpacked ([IO.Path]::GetFileNameWithoutExtension($archive))
    Copy-Item -Path (Join-Path $packageRoot '*') -Destination $cudaRoot -Recurse -Force
    Write-Host "Verified and installed $component"
}
& (Join-Path $cudaRoot 'bin\nvcc.exe') --version
if ($LASTEXITCODE -ne 0) { throw 'CUDA compiler verification failed.' }
Write-Host "Local toolkit: $cudaRoot"
