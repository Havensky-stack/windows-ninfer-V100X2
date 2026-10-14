"""Frozen paired GPQA-Diamond cases and resumable native Engine runs."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health_log import HealthLog, durable_json

SOURCE = 'https://modelscope.cn/datasets/AI-ModelScope/gpqa_diamond/resolve/master/train.jsonl'
DEFAULT_WEIGHTS = '/Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer'
BINARY = 'build-v100-tp4/bench/ninfer_v100_gpqa'
PROFILES = {
    'default': DEFAULT_WEIGHTS,
    'long': DEFAULT_WEIGHTS,
    'nvfp4': '/Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer',
    'q4': '/Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer',
}
SETTINGS = ('benchmark', 'source', 'cases_path', 'prompt_tokens', 'capacity', 'kv',
    'tp', 'drafts', 'chunk', 'greedy', 'thinking', 'reasoning_effort', 'prefix_reuse',
    'max_output_tokens')


def prepare(root, weights):
    raw = root / 'dataset.json'
    if not raw.exists():
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            rows = [json.loads(line) for line in response if line.strip()]
        if len(rows) != 198:
            raise ValueError('expected all 198 GPQA-Diamond questions')
        # Answers/explanations are retained only in the local source, never in prompts.
        durable_json(raw, rows)
    else:
        rows = json.loads(raw.read_text())
    questions = []
    for index, row in enumerate(rows):
        question = row['Question'].strip()
        choices = [row[f'Incorrect Answer {i}'].strip() for i in range(1, 4)]
        choices.append(row['Correct Answer'].strip())
        seed = int.from_bytes(hashlib.sha256(question.encode()).digest()[:8], 'big')
        random.Random(seed).shuffle(choices)
        answer = 'ABCD'[choices.index(row['Correct Answer'].strip())]
        prompt = (
            "Answer the following multiple choice question. The last line of your response "
            "should be of the following format: 'ANSWER: [LETTER]' (without quotes) where "
            "[LETTER] is one of A, B, C, D. Think step by step before answering.\n\n"
            + question + '\n\n' + '\n'.join(f'{letter}) {choice}' for letter, choice in zip('ABCD', choices))
        )
        questions.append({'id': index, 'question': question, 'choices': choices,
            'answer': answer, 'domain': row.get('High-level domain', 'unknown'), 'prompt': prompt})
    # Freeze one data-independent shuffled order for both backends and every resume.
    random.Random(20261008).shuffle(questions)
    durable_json(root / 'questions.json', {'source': SOURCE, 'questions': questions})
    subprocess.run([BINARY, 'prepare', weights, str(root / 'questions.json'),
        'profiles/bench/v100-code-85000-iommu-pt.ids', str(root / 'cases.json')], check=True)


def run(root, profile, count, outputs, capacity):
    command = [BINARY, 'run', PROFILES[profile], str(root / 'cases.json'),
        str(root / (profile + '.json')), profile, str(count), str(outputs), str(capacity)]
    name = profile + '-' + str(time.time_ns())
    with HealthLog(root / (name + '-health')) as health:
        durable_json(root / (name + '.command.json'), command)
        with (root / (name + '.log')).open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            health.attach_pid(process.pid)
            code = process.wait()
        health.event('case_done', returncode=code)
        print((root / (name + '.log')).read_text(), flush=True)
        if code:
            raise subprocess.CalledProcessError(code, command)


def compare(a, b):
    paired = sorted(a.keys() & b.keys())
    changes = []
    lost = gained = 0
    for i in paired:
        x, y = a[i], b[i]
        if x['answer'] != y['answer']:
            raise ValueError('paired question answer differs')
        lost += bool(x['correct'] and not y['correct'])
        gained += bool(y['correct'] and not x['correct'])
        if x['prediction'] != y['prediction']:
            changes.append({'id': i, 'reference': x['prediction'], 'candidate': y['prediction'],
                'answer': x['answer'], 'reference_truncated': x['truncated'],
                'candidate_truncated': y['truncated']})
    n = len(paired)
    discordant = lost + gained
    pvalue = min(1, 2 * sum(math.comb(discordant, k) for k in range(min(lost, gained) + 1))
        / 2 ** discordant) if discordant else 1.0
    return {'paired_questions': n, 'reference_correct_candidate_wrong': lost,
        'reference_wrong_candidate_correct': gained,
        'delta_percentage_points': 100 * (gained - lost) / n if n else None,
        'mcnemar_exact_two_sided_p': pvalue, 'changed_answers': changes}


def analyze(data, answers):
    rows = {}
    for p, report in data.items():
        if report['artifact'] != PROFILES[p] or report['profile'] != p:
            raise ValueError('artifact/profile mismatch: ' + p)
        indexed = {}
        for r in report['runs']:
            i = r['id']
            if i in indexed or i not in answers or r['answer'] != answers[i]:
                raise ValueError('duplicate/unknown question or changed answer: ' + p)
            if r['correct'] != (r['prediction'] == r['answer']):
                raise ValueError('inconsistent correctness flag: ' + p)
            if r['prompt_tokens'] != 85000 or r['reused_tokens'] != 0:
                raise ValueError('unexpected occupancy or prefix reuse: ' + p)
            indexed[i] = r
        rows[p] = indexed
    reference = next(iter(data.values()))
    for report in data.values():
        for key in SETTINGS:
            if report[key] != reference[key]:
                raise ValueError('comparison settings differ at ' + key)
    common = set.intersection(*(set(r) for r in rows.values()))
    result = {'benchmark': reference['benchmark'], 'common_questions': len(common),
        'total_questions': len(answers), 'complete': all(set(r) == set(answers) for r in rows.values()),
        'profiles': {}, 'comparisons': {}}
    for p in rows:
        selected = list(rows[p].values())
        n = len(selected)
        correct = sum(r['correct'] for r in selected)
        seconds = sum(r['prefill_seconds'] + r['decode_seconds'] for r in selected)
        completed_answers = [r for r in selected if not r['truncated'] and not r['parse_failed']]
        result['profiles'][p] = {'artifact': data[p]['artifact'], 'correct': correct, 'total': n,
            'accuracy_percent': 100 * correct / n if n else None,
            'common_correct': sum(rows[p][i]['correct'] for i in common),
            'common_accuracy_percent': 100 * sum(rows[p][i]['correct'] for i in common) / len(common)
                if common else None,
            'truncated': sum(r['truncated'] for r in selected),
            'parse_failed': sum(r['parse_failed'] for r in selected),
            'thinking_not_ended': sum(not r['thinking_ended'] for r in selected),
            'completed_answers': len(completed_answers),
            'completed_answer_accuracy_percent': 100 * sum(r['correct'] for r in completed_answers)
                / len(completed_answers) if completed_answers else None,
            'output_tokens': sum(r['output_tokens'] for r in selected),
            'mean_question_seconds': seconds / n if n else None}
    for a, b in itertools.combinations(rows, 2):
        result['comparisons'][a + ' -> ' + b] = compare(rows[a], rows[b])
    return result


def summarize(root, profiles):
    questions = json.loads((root / 'questions.json').read_text())['questions']
    answers = {q['id']: q['answer'] for q in questions}
    data = {p: json.loads((root / (p + '.json')).read_text()) for p in profiles
        if (root / (p + '.json')).exists()}
    if not data:
        raise ValueError('no saved results')
    result = analyze(data, answers)
    result['expected_profiles'] = profiles
    result['complete'] &= len(data) == len(profiles)
    durable_json(root / 'summary.json', result)
    print(json.dumps(result, indent=2))
    return result


def matrix(root, profiles, total, batch, outputs, capacity, prepared_root):
    for name in ('cases.json', 'questions.json'):
        destination = root / name
        if not destination.exists():
            destination.symlink_to((prepared_root / name).resolve())
    subprocess.run([BINARY, 'frontends', DEFAULT_WEIGHTS,
        *dict.fromkeys(PROFILES[p] for p in profiles)], check=True)
    questions = json.loads((root / 'questions.json').read_text())['questions']
    if total > len(questions) or total < 1 or batch < 1:
        raise ValueError('invalid question count or batch size')
    for limit in range(min(batch, total), total + batch, batch):
        target = {q['id'] for q in questions[:min(limit, total)]}
        for p in profiles:
            path = root / (p + '.json')
            report = json.loads(path.read_text()) if path.exists() else {'runs': []}
            saved = {r['id'] for r in report['runs']}
            missing = target - saved
            if missing:
                run(root, p, len(missing), outputs, capacity)
            summarize(root, profiles)
        if limit >= total:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run', 'summary', 'matrix'])
    parser.add_argument('--root', type=Path, default=Path('profiles/bench/quasar-gpqa-85k'))
    parser.add_argument('--profile', choices=PROFILES)
    parser.add_argument('--profiles', choices=PROFILES, nargs='+', default=list(PROFILES))
    parser.add_argument('--count', type=int, default=198)
    parser.add_argument('--max-output', type=int, default=32768)
    parser.add_argument('--questions', type=int, default=198)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--weights', default=DEFAULT_WEIGHTS)
    parser.add_argument('--capacity', type=int, default=180000)
    parser.add_argument('--prepared-root', type=Path, default=Path('profiles/bench/quasar-gpqa-85k'))
    args = parser.parse_args()
    os.environ['LD_LIBRARY_PATH'] = str(Path('build/_deps/install/lib').resolve()) + \
        ':/usr/local/cuda-12.8/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare':
        prepare(args.root, args.weights)
    elif args.stage == 'run':
        if args.profile is None:
            parser.error('--profile is required for run')
        run(args.root, args.profile, args.count, args.max_output, args.capacity)
    elif args.stage == 'matrix':
        matrix(args.root, args.profiles, args.questions, args.batch_size, args.max_output,
            args.capacity, args.prepared_root)
    else:
        summarize(args.root, args.profiles)


if __name__ == '__main__':
    main()
