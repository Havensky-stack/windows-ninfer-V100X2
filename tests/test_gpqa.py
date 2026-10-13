import copy
import unittest

from tools.v100.bench_gpqa import analyze, compare, PROFILES


def row(identifier, answer, prediction):
    return {'id': identifier, 'answer': answer, 'prediction': prediction,
        'correct': answer == prediction, 'truncated': False, 'parse_failed': not prediction,
        'thinking_ended': True, 'prompt_tokens': 85000, 'reused_tokens': 0,
        'output_tokens': 100, 'prefill_seconds': 64.0, 'decode_seconds': 1.0}


def report(profile, rows):
    return {'artifact': PROFILES[profile], 'profile': profile, 'runs': rows,
        'benchmark': 'GPQA-Diamond / 85K unrelated-code occupancy variant',
        'source': 'frozen-dataset', 'cases_path': 'frozen-cases', 'prompt_tokens': 85000,
        'capacity': 180000, 'kv': 'int8-group64', 'tp': 2, 'drafts': 4, 'chunk': 2560,
        'greedy': True, 'thinking': True, 'reasoning_effort': 'xhigh',
        'prefix_reuse': False, 'max_output_tokens': 32768}


class GpqaSummaryTest(unittest.TestCase):
    def test_paired_gain_loss_and_changed_wrong_answers(self):
        a = {i: row(i, 'A', p) for i, p in enumerate(['A', 'B', 'B', 'B'])}
        b = {i: row(i, 'A', p) for i, p in enumerate(['C', 'A', 'A', 'C'])}
        result = compare(a, b)
        self.assertEqual(result['paired_questions'], 4)
        self.assertEqual(result['reference_correct_candidate_wrong'], 1)
        self.assertEqual(result['reference_wrong_candidate_correct'], 2)
        self.assertEqual(result['delta_percentage_points'], 25)
        self.assertEqual(len(result['changed_answers']), 4)
        self.assertEqual(result['mcnemar_exact_two_sided_p'], 1)

    def test_truncation_is_in_full_score_and_separate_from_completed_answers(self):
        truncated = row(1, 'B', '')
        truncated.update(truncated=True, thinking_ended=False)
        data = {'default': report('default', [row(0, 'A', 'A'), truncated]),
            'long': report('long', [row(0, 'A', 'B')])}
        result = analyze(data, {0: 'A', 1: 'B'})
        self.assertFalse(result['complete'])
        self.assertEqual(result['common_questions'], 1)
        self.assertEqual(result['profiles']['default']['accuracy_percent'], 50)
        self.assertEqual(result['profiles']['default']['completed_answer_accuracy_percent'], 100)
        self.assertEqual(result['profiles']['default']['truncated'], 1)
        self.assertEqual(result['profiles']['default']['thinking_not_ended'], 1)

    def test_inconsistent_settings_or_scoring_are_rejected(self):
        data = {'default': report('default', [row(0, 'A', 'A')]),
            'long': report('long', [row(0, 'A', 'A')])}
        for mutate in (
            lambda d: d['long'].update(chunk=1024),
            lambda d: d['long']['runs'][0].update(correct=False),
            lambda d: d['long']['runs'].append(row(0, 'A', 'A')),
            lambda d: d['long']['runs'][0].update(reused_tokens=100),
        ):
            changed = copy.deepcopy(data)
            mutate(changed)
            with self.assertRaises(ValueError):
                analyze(changed, {0: 'A'})


if __name__ == '__main__':
    unittest.main()
