# Configure first with build.ps1 -Tests -PythonExecutable <Python 3.11 executable>.
param(
    [string]$BuildDirectory = 'build-windows',
    [switch]$Upstream,
    [switch]$SkipBuild,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-toolchain.ps1')
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$buildDir = [IO.Path]::GetFullPath((Join-Path $repoRoot $BuildDirectory))
$targets = @('ninfer_artifact_reader_test', 'ninfer_artifact_materialization_test',
    'ninfer_media_decode_test', 'ninfer_allreduce_test', 'ninfer_ram_kv_test',
    'ninfer_serve_options_test', 'ninfer_request_log_test',
    'ninfer_openai_schema_test', 'ninfer_anthropic_schema_test')
$extraTests = @()
$supportTargets = @()
if ($Upstream) {
    $targets += @('ninfer_public_api_test', 'ninfer_cli_options_test', 'ninfer_responses_schema_test',
        'ninfer_device_test', 'ninfer_decode_graph_test',
        'ninfer_response_store_test', 'ninfer_tool_call_parser_test', 'ninfer_http_error_handler_test',
        'ninfer_sampling_defaults_test', 'ninfer_concurrent_executor_test', 'ninfer_kv_cache_test',
        'ninfer_state_store_test', 'ninfer_gdn_replay_records_test', 'ninfer_tensor_slice_test',
        'ninfer_qwen3_8_27b_dflash2_v3_load_plan_test', 'ninfer_quasar_load_plan_test',
        'ninfer_host_vision_test', 'ninfer_rmsnorm_test', 'ninfer_gated_rmsnorm_test',
        'ninfer_gated_delta_net_test', 'ninfer_gated_delta_net_replay_record_test',
        'ninfer_gdn_gating_test', 'ninfer_gdn_gating_proj_test', 'ninfer_gdn_replay_fold_test',
        'ninfer_bench_support_test',
        'ninfer_argmax_test', 'ninfer_proposal_argmax_test',
        'ninfer_nvfp4_input_sm70_test', 'ninfer_nvfp4_row_add_sm70_test',
        'ninfer_linear_w8_prepacked_sm70_test', 'ninfer_linear_w8_volta_decode_test',
        'ninfer_linear_nvfp4_a16_test', 'ninfer_linear_swiglu_nvfp4_test',
        'ninfer_attention_headlocal_tp4_test')
    # The unfiltered GDN fixtures include FP8 A8 and legacy split-storage profiles.
    # Compile them, but qualify this Windows Coder390 route with their NVFP4 cases.
    $supportTargets += @('ninfer_attention_headlocal_test', 'ninfer_gdn_projections_split_test',
        'ninfer_gdn_input_proj_conv_snapshot_test', 'ninfer_gdn_input_proj_conv_record_test')
    $extraTests += 'ninfer_attention_tp4_geometry_test'
    $extraTests += @('prefill', 'decode1', 'decode4', 'decode5', 'decode6', 'boundaries', 'short') |
        ForEach-Object { "ninfer_attention_tp2_$_`_test" }
    $extraTests += @('ninfer_nvfp4_gdn_projections_split_test',
        'ninfer_nvfp4_gdn_input_proj_conv_snapshot_test', 'ninfer_nvfp4_gdn_input_proj_conv_record_test')
}
if (-not $SkipBuild) {
    & cmake --build $buildDir --parallel 4 --target @targets @supportTargets ninfer_qwen3_6_frontend_test
    if ($LASTEXITCODE -ne 0) { throw 'Focused test build failed.' }
}
$reportDir = Join-Path $buildDir 'windows-qualification'
New-Item -ItemType Directory -Force $reportDir | Out-Null
$active = [Diagnostics.Stopwatch]::StartNew()
$failures = @()
foreach ($target in ($targets + $extraTests)) {
    $report = Join-Path $reportDir "$target.xml"
    if ($Resume -and (Test-Path $report)) {
        [xml]$previous = Get-Content -LiteralPath $report -Raw
        if ([int]$previous.testsuite.tests -gt 0 -and [int]$previous.testsuite.skipped -eq 0 -and
            $previous.testsuite.failures -eq '0' -and
            (-not $previous.testsuite.errors -or $previous.testsuite.errors -eq '0')) {
            Write-Host "Retaining saved result: $target"
            continue
        }
    }
    # The complete attention oracle evaluates every key at 85K occupancy on the CPU.
    $caseTimeout = if ($target -like 'ninfer_attention_*') { 450 } else { 150 }
    # Reserve the entire case timeout before the ten-minute active ceiling.
    if ($active.Elapsed.TotalSeconds + $caseTimeout -ge 540) {
        Write-Host 'GPU campaign cooldown: 180 seconds at a completed-test boundary.'
        1..3 | ForEach-Object { Start-Sleep -Seconds 60 }
        $active.Restart()
    }
    & ctest --test-dir $buildDir --output-on-failure --no-tests=error --timeout $caseTimeout -j 1 `
        --output-junit $report -R "^$target`$"
    if ($LASTEXITCODE -ne 0) { $failures += $target }
}
# The full frontend suite requires upstream's separate official tokenizer fixture.
# This focused case uses the checked-in Coder390 template and independent Jinja outputs.
& (Join-Path $buildDir 'tests\ninfer_qwen3_6_frontend_test.exe') --coder390-template
if ($LASTEXITCODE -ne 0) { throw 'Coder390 template parity failed.' }
if ($failures.Count) { throw "Failed upstream checks: $($failures -join ', ')" }
