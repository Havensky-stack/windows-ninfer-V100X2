"""Launcher memory estimates, observed progress and portable named configurations."""

from dataclasses import replace
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.v100.launcher_resources import GIB, MIB, LoadStage, MemoryOptions, estimate_memory, query_gpu_memory
from tools.v100.launcher_configs import ConfigurationLibrary


class Resources(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'fixture.ninfer'
        items = [('text/token_embedding', 1000, [10, 50]), ('text/output_head', 1000, [10, 50]),
                 ('text/layers/0/gdn/convolution', 80, [4, 10]), ('text/final_norm', 20, [10]),
                 ('mtp/input_projection', 400, [10, 20]), ('text/draft_head', 200, [10, 10]),
                 ('text/draft_head_token_ids', 16, [4]), ('vision/weight', 200, [10, 10]),
                 ('dflash2/weight', 600, [10, 30])]
        self.objects = [dict(name=name, kind='tensor', bytes=size, shape=shape) for name, size, shape in items]
        self.write_directory(2)

    def write_directory(self, version):
        if version == 2:
            data = dict(identity=dict(model_id='qwen3.8-27b', weights_id='nvfp4'), objects=self.objects)
        else:
            objects = [dict(item) for item in self.objects]
            for item in objects:
                item['id'] = item.pop('name')
            data = dict(metadata=dict(name='qwen3.8-27b'), objects=objects)
        encoded = json.dumps(data).encode()
        self.path.write_bytes(struct.pack('<8sQ', b'NINFER\0' + bytes([version]), len(encoded)) +
                              (bytes(16) if version == 3 else b'') + encoded + bytes(4096))

    def test_replicated_embedding_optional_weights_and_vision_rank(self):
        options = MemoryOptions()
        plain = estimate_memory(self.path, options)
        self.assertEqual([e.weights for e in plain], [1876, 1876])
        vision = estimate_memory(self.path, replace(options, vision=True))
        self.assertEqual([e.weights for e in vision], [2076, 1876])
        none = estimate_memory(self.path, replace(options, backend='none'))
        self.assertEqual([e.weights for e in none], [1560, 1560])
        dflash = estimate_memory(self.path, replace(options, backend='dflash', optimized=False))
        self.assertEqual([e.weights for e in dflash], [2160, 2160])
        self.write_directory(3)
        self.assertEqual([e.weights for e in estimate_memory(self.path, options)], [1876, 1876])

    def test_kv_page_rounding_capacity_storage_and_shared_pool(self):
        options = MemoryOptions(backend='none', context=85000, capacity='180000')
        estimate = estimate_memory(self.path, options)[0]
        self.assertEqual(estimate.kv, 180032 * 33792 // 2)
        concurrent = estimate_memory(self.path, replace(options, concurrency=4))[0]
        self.assertEqual(concurrent.kv, estimate.kv)
        self.assertGreater(concurrent.state, estimate.state)
        bf16 = estimate_memory(self.path, replace(options, kv_dtype='bf16'))[0]
        self.assertEqual(bf16.kv, 180032 * 65536 // 2)
        ram = estimate_memory(self.path, replace(options, ram_window=32768))[0]
        self.assertEqual(ram.kv, 32768 * 33792 // 2)
        auto = estimate_memory(self.path, replace(options, capacity='auto'))[0]
        self.assertTrue(auto.auto)
        self.assertEqual(auto.kv, 85056 * 33792 // 2)

    def test_pipeline_endpoints_and_mtp_are_owned_by_primary(self):
        estimates = estimate_memory(self.path, MemoryOptions(tp=1, pp=2))
        self.assertEqual([e.weights for e in estimates], [2636, 1080])
        self.assertGreater(estimates[0].kv, estimates[1].kv)

    def test_gpu_memory_respects_cuda_visible_device_order_and_uuid(self):
        output = '0, GPU-alpha, V100, 16384, 512, 15600\n1, GPU-beta, V100, 16384, 256, 15856\n'
        result = subprocess.CompletedProcess([], 0, output, '')
        with patch('tools.v100.launcher_resources.subprocess.run', return_value=result), \
                patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1,0'}):
            devices = query_gpu_memory()
            self.assertEqual(devices[0].physical, '1')
            self.assertEqual(devices[1].used, 512 * MIB)
        with patch('tools.v100.launcher_resources.subprocess.run', return_value=result), \
                patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-beta'}):
            self.assertEqual(query_gpu_memory()[0].uuid, 'GPU-beta')

    def test_observed_load_progress_handles_chunks_and_does_not_finish_at_upload(self):
        state = LoadStage()
        self.assertFalse(state.feed('[INFO] load weights 12.'))
        self.assertTrue(state.feed('50% 1 GiB / 8 GiB\n'))
        self.assertEqual((state.label, state.percent), ('上传权重', 12.5))
        state.feed('[INFO] load weights 100.00% 8 GiB / 8 GiB\n')
        self.assertIsNone(state.percent)
        self.assertFalse(state.ready)
        state.feed('[INFO] warming up...\n')
        self.assertEqual(state.label, '预热')
        state.feed('[INFO] listening on http://127.0.0.1:8080\n')
        self.assertTrue(state.ready)
        self.assertEqual(state.percent, 100)

    def test_existing_quasar_measurement_matches_weight_and_kv_components(self):
        model = Path('/Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer')
        report = Path(__file__).resolve().parents[1] / 'profiles/bench/quasar-180-20261018/final-published-mtp4-long.json'
        if not model.is_file() or not report.is_file():
            self.skipTest('Local QUASAR artifact and saved measurement required')
        measured = json.loads(report.read_text())['memory']['devices']
        for estimate, actual in zip(estimate_memory(model, MemoryOptions()), measured):
            self.assertLess(abs(estimate.weights - actual['weights_bytes']), MIB)
            self.assertEqual(estimate.kv, actual['kv_pool_bytes'])
            self.assertLess(abs(estimate.low - actual['reserved_bytes']), GIB)


class Configurations(unittest.TestCase):
    def test_named_round_trip_deletion_and_portable_file_exclude_key(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            library = ConfigurationLibrary(path / 'profiles.json')
            values = {'context': 180000, 'vision': False, 'api_key': 'secret'}
            library.save('文本 180K', values)
            library.save('视觉', {'context': 98304, 'vision': True})
            self.assertEqual(set(library.entries()), {'文本 180K', '视觉'})
            exported = path / 'config.json'
            library.export(exported, '文本 180K', values)
            name, settings = library.read_export(exported)
            self.assertEqual((name, settings), ('文本 180K', {'context': 180000, 'vision': False}))
            self.assertNotIn('secret', exported.read_text() + library.path.read_text())
            library.delete('视觉')
            self.assertEqual(list(library.entries()), ['文本 180K'])
            exported.write_text('{"format": "other", "settings": {}}')
            with self.assertRaises(ValueError):
                library.read_export(exported)
            self.assertEqual(list(library.entries()), ['文本 180K'])


if __name__ == '__main__':
    unittest.main()
