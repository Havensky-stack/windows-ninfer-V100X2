# Configure first with build.ps1 -Tests -PythonExecutable <Python 3.11 executable>.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-toolchain.ps1')
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$buildDir = Join-Path $repoRoot 'build-windows'
$targets = @('ninfer_artifact_reader_test', 'ninfer_artifact_materialization_test',
    'ninfer_media_decode_test', 'ninfer_allreduce_test', 'ninfer_ram_kv_test',
    'ninfer_serve_options_test', 'ninfer_request_log_test',
    'ninfer_openai_schema_test', 'ninfer_anthropic_schema_test')
& cmake --build $buildDir --parallel 4 --target @targets ninfer_qwen3_6_frontend_test
if ($LASTEXITCODE -ne 0) { throw 'Focused test build failed.' }
$pattern = '^(' + ($targets -join '|') + ')$'
& ctest --test-dir $buildDir --output-on-failure --timeout 300 -j 1 -R $pattern
if ($LASTEXITCODE -ne 0) { throw 'Focused tests failed.' }
# The full frontend suite requires upstream's separate official tokenizer fixture.
# This focused case uses the checked-in Coder390 template and independent Jinja outputs.
& (Join-Path $buildDir 'tests\ninfer_qwen3_6_frontend_test.exe') --coder390-template
if ($LASTEXITCODE -ne 0) { throw 'Coder390 template parity failed.' }
