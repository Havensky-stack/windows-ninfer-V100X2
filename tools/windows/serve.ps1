param(
    [Parameter(Mandatory)][string]$Artifact,
    [ValidateRange(1024, 262144)][int]$Context = 16384,
    [ValidateRange(64, 1024)][int]$PrefillChunk = 256,
    [ValidateRange(1, 65535)][int]$Port = 8080
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$server = Join-Path $repoRoot 'build-windows\apps\ninfer-serve.exe'
if (-not (Test-Path $server)) { throw 'Run tools/windows/build.ps1 first.' }
$Artifact = (Resolve-Path -LiteralPath $Artifact).Path
& $server $Artifact --host 127.0.0.1 --port $Port `
    --tp 2 --devices '0,1' --max-concurrency 1 `
    --max-context $Context --kv-capacity $Context --kv-dtype int8 --prefill-chunk $PrefillChunk `
    --default-max-tokens 1024 --no-thinking
if ($LASTEXITCODE -ne 0) { throw "NInfer server exited with code $LASTEXITCODE." }
