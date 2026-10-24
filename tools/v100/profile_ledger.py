#!/usr/bin/env python3
"""Summarize one --profile-measured MTP CUDA-Graph run exported by Nsight Systems.

Device activity is reported separately for each GPU. Kernel/copy sums may overlap;
only their interval union is subtracted when reporting time with no GPU activity.
The first graph launch separates prefill (including first-token proposal) from decode.
MTP verify/proposal attribution follows the round-control kernels in each GPU stream.
"""

import argparse
import bisect
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def category(short, full):
    if short.startswith(("allreduce_peer_sum_", "allreduce_packet_")):
        return "communication.peer_sum"
    if "argmax" in short:
        return "token_decision.argmax"
    if short == "Kernel" and "cutlass" in full:
        return "projection.CUTLASS"
    if short.startswith("fp8_volta_qpn"):
        for pattern, label in (("GdnInput", "gdn_input"),
                               ("AttentionInput", "attention_input"),
                               ("Fp32Contiguous", "gate_up")):
            if pattern in full:
                return "projection.FP8." + label
        return "projection.FP8.linear_or_head"
    if "nvfp4" in short and "dequant" not in short:
        return "projection.NVFP4"
    if short.startswith(("w8_", "q4_")):
        return "projection.draft_" + short.split("_")[0]
    if "dequant" in short:
        return "projection.weight_decode"
    if "scale_rows" in short or "scale_gate_up" in short:
        return "projection.row_scale"
    if "bf16_to_fp16" in short or short == "qpn_cast_activation":
        return "projection.activation_cast"
    if "flash_attn_ext" in short:
        return "attention.prefill_core"
    if ("gqa_attention_small_t_tc" in short or "attention_decode_i8_tiled" in short
            or ("gqa_attention_" in short and "partial_i8" in short)):
        return "attention.decode_partial"
    if "attention_small_t_reduce" in short or "flash_attn_stream_k_fixup" in short:
        return "attention.reduce"
    if "volta_flash" in short or "gqa_attention_prefill" in short:
        return "attention.prepare_KV_mask"
    if "recurrent" in short:
        return "GDN.recurrence_or_QK_prepare"
    if "gdn_gating" in short:
        return "GDN.control_projection"
    if "conv" in short:
        return "GDN.convolution"
    if "rmsnorm" in short:
        return "normalization"
    if "residual_add" in short:
        return "residual_add"
    if "silu" in short or "swiglu" in short or "sigmoid_gate" in short:
        return "activation_and_gate"
    if "rope" in short:
        return "RoPE"
    if "embed" in short:
        return "embedding"
    return "split_sampling_and_control"


def union_ns(intervals):
    total = 0
    end = -1
    for lo, hi in sorted(intervals):
        if hi > end:
            total += hi - max(lo, end)
            end = hi
    return total


def summarize(connection):
    names = dict(connection.execute("SELECT id,value FROM StringIds"))
    runtime = list(connection.execute(
        "SELECT start,end,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"))
    launches = [lo for lo, hi, name in runtime if names[name].startswith("cudaGraphLaunch")]
    if not launches:
        raise ValueError("expected measured CUDA-Graph decode launches")
    split = launches[0]
    kinds = {row[0]: row[2] for row in connection.execute("SELECT * FROM ENUM_CUDA_MEMCPY_OPER")}
    kernels = list(connection.execute(
        "SELECT start,end,deviceId,shortName,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"))
    copies = list(connection.execute(
        "SELECT start,end,deviceId,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY"))
    memsets = list(connection.execute(
        "SELECT start,end,deviceId,bytes FROM CUPTI_ACTIVITY_KIND_MEMSET"))
    devices = sorted({row[2] for row in kernels})
    first = min(row[0] for row in kernels + copies + memsets)
    last = max(row[1] for row in kernels + copies + memsets)
    result = {"graph_launches": len(launches), "devices": {}}
    result["all_devices"] = {}
    for phase, lower, upper in (("prefill", first, split), ("decode", split, last)):
        intervals = [(lo, hi) for lo, hi, *_ in kernels + copies + memsets
                     if lower <= lo < upper]
        result["all_devices"][phase] = {
            "timeline_seconds": (upper - lower) / 1e9,
            "no_activity_seconds": (upper - lower - union_ns(intervals)) / 1e9,
        }
    for device in devices:
        dk = [row for row in kernels if row[2] == device]
        dc = [row for row in copies if row[2] == device]
        dm = [row for row in memsets if row[2] == device]
        markers = sorted((lo, "verify" if names[short].startswith("speculative_prepare_verify")
                          else "proposal")
                         for lo, hi, dev, short, full in dk if lo >= split and
                         names[short] in ("speculative_prepare_verify_inputs_kernel",
                                          "mtp_prepare_next_round_kernel"))
        marker_times = [lo for lo, label in markers]
        phases = {}
        for phase, lower, upper in (("prefill", first, split), ("decode", split, last)):
            rows = [row for row in dk if lower <= row[0] < upper]
            transfers = [row for row in dc if lower <= row[0] < upper]
            fills = [row for row in dm if lower <= row[0] < upper]
            grouped = defaultdict(lambda: [0, 0])
            categories = defaultdict(int)
            mtp = defaultdict(int)
            for lo, hi, dev, short, full in rows:
                name = names[full]
                group = category(names[short], name)
                grouped[(group, name)][0] += 1
                grouped[(group, name)][1] += hi - lo
                categories[group] += hi - lo
                index = bisect.bisect_right(marker_times, lo) - 1
                if phase == "decode" and index >= 0:
                    mtp[markers[index][1]] += hi - lo
            activity = [(lo, hi) for lo, hi, *_ in rows + transfers + fills]
            active = union_ns(activity)
            copy_groups = defaultdict(lambda: [0, 0, 0])
            for lo, hi, dev, kind, size in transfers:
                copy_groups[kinds[kind]][0] += 1
                copy_groups[kinds[kind]][1] += hi - lo
                copy_groups[kinds[kind]][2] += size
            phases[phase] = {
                "timeline_seconds": (upper - lower) / 1e9,
                "kernel_seconds": sum(hi - lo for lo, hi, *_ in rows) / 1e9,
                "activity_union_seconds": active / 1e9,
                "no_activity_seconds": (upper - lower - active) / 1e9,
                "kernel_count": len(rows),
                "categories_seconds": dict(sorted(((k, v / 1e9) for k, v in categories.items()),
                                                   key=lambda row: -row[1])),
                "mtp_kernel_seconds": {k: v / 1e9 for k, v in mtp.items()},
                "copies": {k: {"count": v[0], "seconds": v[1] / 1e9, "bytes": v[2]}
                           for k, v in copy_groups.items()},
                "memset_seconds": sum(hi - lo for lo, hi, *_ in fills) / 1e9,
                "kernels": [{"category": group, "name": name, "calls": count,
                             "seconds": ns / 1e9, "mean_us": ns / count / 1e3}
                            for (group, name), (count, ns) in
                            sorted(grouped.items(), key=lambda item: -item[1][1])],
            }
        assert sum(p["kernel_count"] for p in phases.values()) == len(dk)
        result["devices"][str(device)] = phases
    api = defaultdict(lambda: [0, 0])
    for lo, hi, name in runtime:
        api[names[name]][0] += 1
        api[names[name]][1] += hi - lo
    result["host_api"] = {k: {"calls": v[0], "seconds": v[1] / 1e9}
                          for k, v in sorted(api.items(), key=lambda item: -item[1][1])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bench = json.loads(args.benchmark.read_text())
    if len(bench["tests"]) != 1 or len(bench["tests"][0]["reps"]) != 1:
        parser.error("the associated measured benchmark must contain one test and one repetition")
    connection = sqlite3.connect(args.sqlite.resolve().as_uri() + "?mode=ro", uri=True)
    with connection:
        report = summarize(connection)
    report["source"] = {"trace": str(args.sqlite), "benchmark": str(args.benchmark)}
    report["environment"] = bench["environment"]
    report["load"] = bench["load"]
    rep = bench["tests"][0]["reps"][0]
    report["request"] = {k: v for k, v in rep.items() if k != "generation"}
    report["notes"] = [
        "Profiler timings include tracing overhead; use unprofiled repeats for throughput.",
        "Load/upload is outside the trace. Upload is part of load, not additive.",
        "First graph launch separates prefill/initial drafts from measured decode.",
        "Device timelines share bounds from first to last captured GPU activity.",
        "Kernel and copy sums are not wall time; devices run concurrently.",
        "No-activity intervals include scheduling, synchronization and peer waits; they are not all PCIe transfer time.",
        "Host API time can include waiting and is not additive to device time.",
        "Driver-staged copy records do not expose all PCIe/host staging time; do not infer physical bandwidth from their bytes/duration.",
    ]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for device, phases in report["devices"].items():
        for phase, row in phases.items():
            print(f"GPU {device} {phase}: timeline={row['timeline_seconds']:.6f}s "
                  f"kernels={row['kernel_seconds']:.6f}s "
                  f"no_activity={row['no_activity_seconds']:.6f}s")


if __name__ == "__main__":
    main()
