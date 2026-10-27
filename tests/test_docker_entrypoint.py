"""Container pull/cache and launcher handoff checks without CUDA or large downloads."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / 'docker/entrypoint.py'
spec = importlib.util.spec_from_file_location('ninfer_docker_entrypoint', ENTRYPOINT)
entrypoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrypoint)


class ContainerLaunch(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_hub_urls_preserve_revision_and_select_native_file(self):
        for source in (entrypoint.DEFAULT_MODEL,
                       entrypoint.DEFAULT_MODEL.replace('/resolve/', '/blob/'),
                       f'https://huggingface.co/{entrypoint.DEFAULT_REPO}/tree/v3'):
            with self.subTest(source=source):
                repo, revision, filename = entrypoint.hub_source(source, None, None)
                self.assertEqual((repo, revision), (entrypoint.DEFAULT_REPO, 'v3'))
                if '/tree/' not in source:
                    self.assertEqual(filename, entrypoint.DEFAULT_FILENAME)
        self.assertEqual(entrypoint.hub_source('other/model', None, None),
                         ('other/model', 'main', None))
        self.assertEqual(entrypoint.hub_source(
            'https://huggingface.co/other/model/resolve/refs%2Fpr%2F2/sub/file.ninfer', None, None),
            ('other/model', 'refs/pr/2', 'sub/file.ninfer'))
        for filename in ('model.safetensors', 'model.gguf', '../model.ninfer'):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                entrypoint.hub_source('other/model', None, filename)

    def test_hub_ambiguous_repository_requires_explicit_filename(self):
        download = Mock(return_value=self.directory / 'chosen.ninfer')
        api = Mock()
        api.list_repo_files.return_value = ['README.md', 'a.ninfer', 'b.ninfer']
        client = SimpleNamespace(HfApi=Mock(return_value=api), hf_hub_download=download)
        with patch.dict(sys.modules, {'huggingface_hub': client}):
            with self.assertRaisesRegex(ValueError, '--filename'):
                entrypoint.pull_model('other/model', self.directory)
            download.assert_not_called()
            entrypoint.pull_model('other/model', self.directory, 'release', 'b.ninfer')
            self.assertEqual(download.call_args.kwargs['revision'], 'release')
            self.assertEqual(download.call_args.kwargs['filename'], 'b.ninfer')
            api.list_repo_files.return_value = ['README.md', entrypoint.DEFAULT_FILENAME]
            entrypoint.pull_model(f'https://huggingface.co/{entrypoint.DEFAULT_REPO}/tree/v3',
                                  self.directory)
            self.assertEqual(download.call_args.kwargs['local_dir'], self.directory / 'quasar-v3')

    def test_http_download_resume_cache_and_separate_sources(self):
        requests = []
        payload = b'fixture native artifact\x00' * 1024

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                offset = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
                requests.append((self.path, offset))
                data = payload if self.path.startswith('/one/') else payload + b'other'
                self.send_response(206 if offset else 200)
                self.send_header('Content-Length', str(len(data) - offset))
                if offset:
                    self.send_header('Content-Range', f'bytes {offset}-{len(data)-1}/{len(data)}')
                self.end_headers()
                self.wfile.write(data[offset:])

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        address = f'http://127.0.0.1:{server.server_port}'
        source = address + '/one/model.ninfer'
        artifact = entrypoint.pull_model(source, self.directory)
        self.assertEqual(artifact.read_bytes(), payload)
        self.assertEqual(entrypoint.pull_model(source, self.directory), artifact)
        self.assertEqual(len(requests), 1)
        artifact.unlink()
        artifact.with_suffix('.ninfer.partial').write_bytes(payload[:137])
        self.assertEqual(entrypoint.pull_model(source, self.directory).read_bytes(), payload)
        self.assertEqual(requests[-1][1], 137)
        other = entrypoint.pull_model(address + '/two/model.ninfer', self.directory)
        self.assertNotEqual(artifact, other)
        self.assertEqual(other.read_bytes(), payload + b'other')
        self.assertEqual(artifact.read_bytes(), payload)

    @unittest.skipUnless(os.name == 'posix', 'Docker TUI handoff requires POSIX terminal support')
    def test_local_model_enters_real_tui_command_path_and_restores_settings(self):
        artifact = self.directory / 'model with spaces.ninfer'
        artifact.touch()
        config = self.directory / 'config/ninfer/tui.json'
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({'context': 98304, 'thinking': False}))
        result = subprocess.run([sys.executable, str(ENTRYPOINT), '--model', str(artifact),
            '--binary', sys.executable, '--print-command'],
            env={**os.environ, 'XDG_CONFIG_HOME': str(self.directory / 'config')},
            capture_output=True, text=True, check=True)
        command = shlex.split(result.stdout.splitlines()[-1])
        self.assertEqual(command[:2], [str(Path(sys.executable).resolve()), str(artifact)])
        for key, value in (('--max-context', '98304'), ('--host', '0.0.0.0'),
                           ('--draft-tokens', '4'), ('--request-log-jsonl', '/data/logs/requests.jsonl')):
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertIn('--no-thinking', command)
        result = subprocess.run([sys.executable, str(ENTRYPOINT), '--model', str(artifact),
            '--print-command', '--no-settings', '--host', '127.0.0.1',
            '--log', str(self.directory / 'override.jsonl')],
            capture_output=True, text=True, check=True)
        command = shlex.split(result.stdout.splitlines()[-1])
        self.assertEqual(command[command.index('--host') + 1], '127.0.0.1')
        self.assertEqual(command[command.index('--request-log-jsonl') + 1],
                         str(self.directory / 'override.jsonl'))

    def test_noninteractive_start_fails_before_download_and_pull_only_works(self):
        result = subprocess.run([sys.executable, str(ENTRYPOINT)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('docker run -it', result.stderr)
        artifact = self.directory / 'model.ninfer'
        artifact.touch()
        result = subprocess.run([sys.executable, str(ENTRYPOINT), '--model', str(artifact),
                                 '--pull-only'], capture_output=True, text=True, check=True)
        self.assertIn(str(artifact), result.stdout)


if __name__ == '__main__':
    unittest.main()
