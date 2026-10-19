#!/usr/bin/env python3
"""Measure the first nonempty SSE text delta over HTTP on a fully cached 300-token prompt.

Uses the standard library. The server must allow prefix reuse and have one idle request lane.
Reports fresh and persistent connections, with both chat-history preservation settings.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import statistics
import time
from urllib.parse import urlsplit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:6198')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=15)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('--repetitions must be positive')
    endpoint = urlsplit(args.base_url)
    if endpoint.scheme not in ('http', 'https') or not endpoint.hostname:
        parser.error('--base-url must be an HTTP(S) URL')
    connection_type = (http.client.HTTPSConnection if endpoint.scheme == 'https'
                       else http.client.HTTPConnection)

    def connect():
        return connection_type(endpoint.hostname, endpoint.port, timeout=120)

    def count(items):
        body = {'model': args.model, 'messages': [{'role': 'user', 'content': prompt(items)}],
                'thinking': {'type': 'disabled'}, 'preserve_thinking': True}
        connection = connect()
        try:
            connection.request('POST', endpoint.path.rstrip('/') + '/v1/messages/count_tokens',
                               json.dumps(body), {'Content-Type': 'application/json'})
            response = connection.getresponse()
            data = response.read()
            if response.status != 200:
                raise RuntimeError(f'count_tokens HTTP {response.status}: {data!r}')
            return json.loads(data)['input_tokens']
        finally:
            connection.close()

    def prompt(items):
        return ('Continue this numbered list with short English sentences. Start with One.\n'
                + ' item' * items)

    low, high = 0, 300
    while low < high:
        middle = (low + high) // 2
        if count(middle) < 300:
            low = middle + 1
        else:
            high = middle
    if count(low) != 300:
        raise RuntimeError('This model cannot form the exact 300-token fixture')
    report = {'base_url': args.base_url, 'model': args.model, 'prompt': prompt(low),
              'prompt_tokens': 300, 'output_limit': 64, 'warmups_per_case': 2,
              'repetitions': args.repetitions, 'rows': [], 'summary': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for preserve in (True, False):
        for reuse in (False, True):
            connection = None
            expected = None
            try:
                for repetition in range(-2, args.repetitions):
                    if connection is None:
                        connection = connect()
                    body = json.dumps({
                        'model': args.model, 'messages': [{'role': 'user', 'content': prompt(low)}],
                        'max_tokens': 64, 'stream': True, 'stream_options': {'include_usage': True},
                        'temperature': 0, 'presence_penalty': 0, 'frequency_penalty': 0,
                        'enable_thinking': False, 'preserve_thinking': preserve,
                    })
                    started = time.perf_counter_ns()
                    connection.request('POST', endpoint.path.rstrip('/') + '/v1/chat/completions',
                                       body, {'Content-Type': 'application/json',
                                              'Accept': 'text/event-stream'})
                    response = connection.getresponse()
                    headers_at = time.perf_counter_ns()
                    if response.status != 200:
                        raise RuntimeError(f'completion HTTP {response.status}: {response.read()!r}')
                    first_event = first_content = usage = finish = None
                    chunks = []
                    done = False
                    while line := response.readline():
                        received = time.perf_counter_ns()
                        if not line.startswith(b'data: '):
                            continue
                        if first_event is None:
                            first_event = received
                        data = line[6:].strip()
                        if data == b'[DONE]':
                            done = True
                            continue
                        event = json.loads(data)
                        if 'error' in event:
                            raise RuntimeError(event)
                        if event.get('usage'):
                            usage = event['usage']
                        for choice in event.get('choices', []):
                            delta = choice.get('delta', {})
                            text = delta.get('content', '') or delta.get('reasoning_content', '')
                            if text:
                                if first_content is None:
                                    first_content = received
                                chunks.append(text)
                            if choice.get('finish_reason'):
                                finish = choice['finish_reason']
                    ended = time.perf_counter_ns()
                    if not done or first_content is None or usage is None:
                        raise RuntimeError('Incomplete SSE response or missing text/usage')
                    digest = hashlib.sha256(''.join(chunks).encode()).hexdigest()
                    observed = (digest, usage['completion_tokens'], finish)
                    if expected is None:
                        expected = observed
                    if observed != expected:
                        raise RuntimeError('Greedy output changed between repetitions')
                    if repetition >= 0:
                        if (usage['prompt_tokens'] != 300 or
                                usage['prompt_tokens_details']['cached_tokens'] != 300):
                            raise RuntimeError(f'Measurement is not a complete 300-token hit: {usage}')
                        report['rows'].append({
                            'preserve_thinking': preserve, 'keep_alive': reuse, 'rep': repetition,
                            'headers_ms': (headers_at - started) / 1e6,
                            'first_event_ms': (first_event - started) / 1e6,
                            'first_content_ms': (first_content - started) / 1e6,
                            'total_ms': (ended - started) / 1e6,
                            'usage': usage, 'sha256': digest, 'finish': finish,
                        })
                    if not reuse:
                        connection.close()
                        connection = None
            finally:
                if connection is not None:
                    connection.close()
            rows = [r for r in report['rows']
                    if r['preserve_thinking'] == preserve and r['keep_alive'] == reuse]
            summary = {'preserve_thinking': preserve, 'keep_alive': reuse}
            for key in ('headers_ms', 'first_event_ms', 'first_content_ms', 'total_ms'):
                values = sorted(r[key] for r in rows)
                summary[key] = {'median': statistics.median(values),
                                'min': values[0], 'max': values[-1]}
            report['summary'].append(summary)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
