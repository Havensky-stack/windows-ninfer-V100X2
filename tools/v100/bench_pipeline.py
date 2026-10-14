#!/usr/bin/env python3
"""Compare startup-fixed TP2 and PP2 on saved identical code-task inputs.

Reports are saved per case. Completed cases are reused only when their command matches.
Cooldowns happen outside request timings, at request boundaries and between parallel layouts.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import statistics
import time

from bench_tp import summarize
from health_log import HealthLog, durable_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=Path("profiles/bench"))
    parser.add_argument("--backends", nargs="+", choices=("tp2", "pp2", "pp2-serial"),
                        default=["tp2", "pp2", "pp2-serial"])
    parser.add_argument("--prompts", nargs="+", type=int,
                        default=[512, 3072, 8192, 16384, 32768, 65536, 85000])
    parser.add_argument("--drafts", nargs="+", type=int, choices=(0, 3), default=[0, 3])
    parser.add_argument("--capacity", type=int, default=180000)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--cooldown", type=int, default=180)
    parser.add_argument("--active-seconds", type=int, default=480)
    args = parser.parse_args()
    if args.repetitions < 1 or args.active_seconds > 600 or args.cooldown < 180:
        parser.error("positive repetitions, <=600 active seconds and >=180 cooldown required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    reports = {}
    active_start = None
    previous_backend = None
    case_estimate = 0.0

    def rest():
        nonlocal active_start
        print(f"Cooldown {args.cooldown}s outside measurements", flush=True)
        end = time.monotonic() + args.cooldown
        while time.monotonic() < end:
            time.sleep(min(1, end - time.monotonic()))
        active_start = None

    for backend in args.backends:
        for drafts in args.drafts:
            for tokens in args.prompts:
                if tokens == 512:
                    corpus = args.corpus_dir / "v100-code-512-p2p.ids"
                elif tokens == 85000:
                    corpus = args.corpus_dir / "v100-code-85000-iommu-pt.ids"
                else:
                    corpus = args.corpus_dir / "p2p_occupancy" / f"code-{tokens}.ids"
                if not corpus.is_file():
                    raise FileNotFoundError(corpus)
                name = f"{backend}-c{args.capacity}-p{tokens}-k{args.prefill_chunk}-mtp{drafts}"
                if args.window:
                    name += f"-ram{args.window}"
                raw = args.output_dir / f"{name}.json"
                parallel = ["--tp", "2"] if backend == "tp2" else [
                    "--pp", "2", "--pipeline-chunks", "1" if backend == "pp2-serial" else "8"]
                command = [str(args.bench.resolve()), "--weights", str(args.weights.resolve()),
                           *parallel, "--devices", "0,1", "--max-ctx", str(args.capacity),
                           "--kv-dtype", "int8", "--prefill-chunk", str(args.prefill_chunk),
                           "--spec", "mtp" if drafts else "none", "--draft-tokens", str(drafts),
                           *(["--lm-head-draft"] if drafts else []),
                           "--corpus", str(corpus.resolve()), "-pg", f"{tokens},{args.output_tokens}",
                           "--warmup", "0", "-r", str(args.repetitions), "--capture-generation",
                           "-o", "json", "--output-file", str(raw.resolve())]
                if args.window:
                    command += ["--ram-kv-window", str(args.window)]
                command_path = raw.with_suffix(".command.json")
                if raw.exists():
                    if json.loads(command_path.read_text()) != command:
                        raise RuntimeError(f"saved case command mismatch: {name}")
                    report = json.loads(raw.read_text())
                else:
                    if previous_backend is None:
                        rest()
                    elif active_start is not None and (previous_backend != backend or
                            time.monotonic() - active_start + max(60, case_estimate) > args.active_seconds):
                        rest()
                    if active_start is None:
                        active_start = time.monotonic()
                    previous_backend = backend
                    print("Running", name, flush=True)
                    durable_json(command_path, command)
                    started = time.monotonic()
                    health_dir = args.output_dir / f"{name}-health-{time.time_ns()}"
                    with HealthLog(health_dir, interval=0.5) as health:
                        health.event("case_start", command=command)
                        with raw.with_suffix(".log").open("w") as log:
                            process = subprocess.Popen(command, env=os.environ.copy(), stdout=log,
                                                       stderr=subprocess.STDOUT)
                            health.attach_pid(process.pid)
                            code = process.wait()
                        health.event("case_end", returncode=code)
                        if code != 0:
                            raise subprocess.CalledProcessError(code, command)
                    case_estimate = max(case_estimate, time.monotonic() - started)
                    report = json.loads(raw.read_text())
                row = summarize(report)
                row.update(backend=backend, drafts=drafts, chunk=args.prefill_chunk, window=args.window,
                           raw_report=raw.name, devices=report["environment"]["devices"],
                           device_memory=report["memory"]["devices"],
                           phase_seconds={phase: statistics.mean(
                               rep["timings"][phase] for rep in report["tests"][0]["reps"])
                               for phase in ("prepare_seconds", "first_token_seconds",
                                             "prefill_seconds", "decode_seconds", "total_seconds")})
                reports[backend, drafts, tokens] = report
                rows.append(row)
                durable_json(args.output_dir / "summary.json", rows)
                print(json.dumps(row), flush=True)
    if active_start is not None:
        rest()
    equivalence = []
    for drafts in args.drafts:
        for tokens in args.prompts:
            wave = reports.get(("pp2", drafts, tokens))
            serial = reports.get(("pp2-serial", drafts, tokens))
            if wave is None or serial is None:
                continue
            pairs = zip(wave["tests"][0]["reps"], serial["tests"][0]["reps"], strict=True)
            equal = all(a["generation"]["token_ids"] == b["generation"]["token_ids"] and
                        a["speculative"] == b["speculative"] for a, b in pairs)
            equivalence.append({"prompt_tokens": tokens, "drafts": drafts,
                                "wave_serial_output_and_speculation_equal": equal})
    durable_json(args.output_dir / "equivalence.json", equivalence)
    if not args.window and any(not row["wave_serial_output_and_speculation_equal"] for row in equivalence):
        raise RuntimeError("PP wave/serial state replay differs; inspect saved reports")


if __name__ == "__main__":
    main()
