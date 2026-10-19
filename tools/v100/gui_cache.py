"""Cache, phase and live draft statistics from the owned server's JSONL log."""

from __future__ import annotations

import json
import math
from pathlib import Path


class CacheMonitor:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.requests = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.latest_prompt_tokens = 0
        self.latest_cached_tokens = 0
        self.live_drafted_tokens = 0
        self.live_accepted_tokens = 0
        self.live_interval_seconds = 0.0
        self.live_prefill_rate = 0.0
        self.live_decode_rate = 0.0
        self.latest_prefill_tokens = 0
        self.latest_decode_tokens = 0
        self.latest_prefill_seconds = None
        self.latest_decode_seconds = None
        self.latest_drafted_tokens = 0
        self.latest_accepted_tokens = 0

    @staticmethod
    def seconds(value):
        return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None

    def poll(self) -> bool:
        """Consume complete records once; leave a partial final line for the next poll."""
        changed = False
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                for _ in range(128):
                    line = stream.readline()
                    if not line.endswith(b"\n"):
                        break
                    self.offset = stream.tell()
                    try:
                        record = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if not isinstance(record, dict):
                        continue
                    if record.get("event") == "throughput":
                        stats = record.get("speculative", {})
                        drafted = stats.get("drafted_tokens") if isinstance(stats, dict) else None
                        accepted = stats.get("accepted_tokens") if isinstance(stats, dict) else None
                        interval = self.seconds(record.get("interval_seconds"))
                        if (type(drafted) is int and type(accepted) is int and
                                0 <= accepted <= drafted and interval is not None and interval > 0):
                            self.live_drafted_tokens = drafted
                            self.live_accepted_tokens = accepted
                            self.live_interval_seconds = interval
                            rates = record.get("throughput_tokens_per_second", {})
                            if isinstance(rates, dict):
                                self.live_prefill_rate = self.seconds(rates.get("prefill")) or 0.0
                                self.live_decode_rate = self.seconds(rates.get("decode")) or 0.0
                            changed = True
                        continue
                    if record.get("event") != "request_done":
                        continue
                    result = record.get("result")
                    if not isinstance(result, dict):
                        continue
                    prompt = result.get("prompt_tokens")
                    cached = result.get("prefix_cache_hit_tokens")
                    if type(prompt) is not int or type(cached) is not int:
                        continue
                    if not 0 <= cached <= prompt:
                        continue
                    self.requests += 1
                    self.latest_prompt_tokens = prompt
                    self.latest_cached_tokens = cached
                    self.prompt_tokens += prompt
                    self.cached_tokens += cached
                    timings = record.get("timings_seconds", {})
                    timings = timings if isinstance(timings, dict) else {}
                    self.latest_prefill_tokens = prompt - cached
                    completion = result.get("completion_tokens", 0)
                    self.latest_decode_tokens = max(completion - 1, 0) if type(completion) is int else 0
                    self.latest_prefill_seconds = self.seconds(timings.get("prefill"))
                    self.latest_decode_seconds = self.seconds(timings.get("decode"))
                    stats = record.get("speculative", {})
                    drafted = stats.get("drafted_tokens", 0) if isinstance(stats, dict) else 0
                    accepted = stats.get("accepted_tokens", 0) if isinstance(stats, dict) else 0
                    self.latest_drafted_tokens = drafted if type(drafted) is int and drafted >= 0 else 0
                    self.latest_accepted_tokens = (accepted if type(accepted) is int and
                                                  0 <= accepted <= self.latest_drafted_tokens else 0)
                    changed = True
        except FileNotFoundError:
            pass  # The server creates its log during startup.
        return changed

    @property
    def latest_rate(self) -> float:
        return self.latest_cached_tokens / self.latest_prompt_tokens if self.latest_prompt_tokens else 0.0

    @property
    def total_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0
