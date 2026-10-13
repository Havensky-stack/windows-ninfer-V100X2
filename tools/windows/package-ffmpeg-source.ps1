param(
    [string]$SourceArchive = (Join-Path $env:LOCALAPPDATA 'vcpkg\downloads\ffmpeg-ffmpeg-n8.1.2.tar.gz'),
    [string]$PortDirectory = (Join-Path $env:LOCALAPPDATA 'vcpkg\registries\git-trees\c40aaa40e075d25ef1e812daa7333778890b6b0d')
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$expected = 'c72f4062aecc16d8b2b1e8678d5efe3af4cfaa0cc7c0997052248f9e499e60c2463acf07877cf3b78b246ce3e8078cb043e8d97e90a6b50d06af32ff7369a788'
if ((Get-FileHash -LiteralPath $SourceArchive -Algorithm SHA512).Hash.ToLowerInvariant() -ne $expected) { throw 'FFmpeg source checksum mismatch.' }
$root = Join-Path $repo 'build-release\FFmpeg-8.1.2-source'
New-Item -ItemType Directory -Force $root, (Join-Path $root 'vcpkg-port') | Out-Null
Copy-Item -LiteralPath $SourceArchive -Destination $root -Force
Copy-Item -Path (Join-Path $PortDirectory '*') -Destination (Join-Path $root 'vcpkg-port') -Recurse -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'launcher\FFMPEG-SOURCE.md') -Destination (Join-Path $root 'README.md') -Force
Copy-Item -LiteralPath (Join-Path $repo 'vcpkg.json'), (Join-Path $PSScriptRoot 'triplets\x64-windows-ninfer.cmake') -Destination $root -Force
Compress-Archive -LiteralPath $root -DestinationPath (Join-Path $repo 'build-release\FFmpeg-8.1.2-source.zip') -CompressionLevel Optimal -Force
