"""Model control HTTP/auth/error/transition behavior, without CUDA or Qt."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import unittest

from tools.v100.launcher_model import ModelControl


class ModelControlTests(unittest.TestCase):
    def setUp(self):
        self.state = 'gpu'
        self.active = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def respond(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def authorized(self):
                if self.headers.get('Authorization') == 'Bearer test-key':
                    return True
                self.respond(401, {'error': {'message': 'missing or invalid API key'}})
                return False

            def do_GET(self):
                if self.authorized():
                    self.respond(200, {'state': owner.state, 'retained_host_bytes': 12345678,
                                       'phase': 'retain_host', 'done': 40, 'total': 100,
                                       'active_requests': owner.active})

            def do_POST(self):
                if not self.authorized():
                    return
                owner.calls.append(self.path)
                if owner.active:
                    self.respond(409, {'error': {'message': 'active inference request'}})
                    return
                owner.state = 'offloading' if self.path.endswith('offload') else 'loading'
                owner.entered.set()
                owner.release.wait(5)
                owner.state = 'ram' if self.path.endswith('offload') else 'gpu'
                self.respond(200, {'state': owner.state, 'retained_host_bytes': 12345678,
                                   'active_requests': 0})

        self.http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.worker = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.worker.start()
        self.model = ModelControl(f'http://127.0.0.1:{self.http.server_port}', 'test-key')
        self.addCleanup(self.close)

    def close(self):
        self.release.set()
        self.http.shutdown()
        self.http.server_close()
        self.worker.join(2)

    def until(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.model.poll()
            if predicate():
                return
            time.sleep(.01)
        self.fail(f'model transition timed out: {self.model.state}, {self.model.error}')

    def test_async_offload_load_with_progress_and_auth(self):
        self.until(lambda: self.model.state == 'gpu')
        self.assertTrue(self.model.action('offload'))
        self.assertTrue(self.entered.wait(2))
        self.assertFalse(self.model.action('offload'))
        self.until(lambda: self.model.phase == 'retain_host')
        self.assertTrue(self.model.busy)
        self.assertEqual((self.model.done, self.model.total), (40, 100))
        self.release.set()
        self.until(lambda: self.model.state == 'ram' and not self.model.busy)
        self.assertEqual(self.model.retained_host_bytes, 12345678)
        self.assertFalse(self.model.action('offload'))
        self.assertTrue(self.model.action('load'))
        self.until(lambda: self.model.state == 'gpu' and not self.model.busy)
        self.assertEqual(self.calls, ['/ninfer/model/offload', '/ninfer/model/load'])

    def test_busy_rejection_recovers_current_state(self):
        self.until(lambda: self.model.state == 'gpu')
        self.active = 1
        self.model.action('offload')
        self.until(lambda: bool(self.model.error))
        self.assertEqual(self.model.error, 'active inference request')
        self.assertFalse(self.model.busy)
        self.until(lambda: self.model.state == 'gpu')
        self.assertEqual(self.model.active_requests, 1)

    def test_wrong_key_never_becomes_ready(self):
        self.model.headers = {'Authorization': 'Bearer invalid'}
        self.model.poll()
        self.until(lambda: not self.model._polling)
        self.assertEqual(self.model.state, 'unknown')
        self.assertFalse(self.model.action('offload'))


if __name__ == '__main__':
    unittest.main()
