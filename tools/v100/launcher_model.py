"""Asynchronous model residency controls shared by the GUI and TUI launchers."""
from __future__ import annotations

import json
import queue
import threading
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


class ModelControl:
    def __init__(self, base_url: str, api_key: str = '') -> None:
        self.url = base_url.rstrip('/') + '/ninfer/model'
        self.headers = {'Authorization': 'Bearer ' + api_key} if api_key else {}
        self.state = 'unknown'
        self.retained_host_bytes = 0
        self.phase = ''
        self.done = self.total = self.active_requests = 0
        self.busy = False
        self.error = ''
        self._results: queue.Queue = queue.Queue()
        self._polling = False
        self._next_poll = 0.0
        self._revision = 0

    def _request(self, action: str, revision: int) -> None:
        try:
            request = Request(self.url + action, headers=self.headers,
                              method='POST' if action else 'GET')
            opener = build_opener(ProxyHandler({}))
            with opener.open(request, timeout=1800 if action else 1.5) as response:
                result = json.load(response)
            if result.get('state') not in ('gpu', 'ram', 'offloading', 'loading'):
                raise ValueError('Invalid model residency response')
            self._results.put((revision, action, result, ''))
        except (OSError, ValueError) as error:
            detail = str(error)
            if isinstance(error, HTTPError):
                try:
                    detail = json.load(error)['error']['message']
                except (KeyError, TypeError, ValueError):
                    pass
            self._results.put((revision, action, None, detail))

    def action(self, action: str) -> bool:
        if self.busy or action not in ('offload', 'load'):
            return False
        if self.state != ('gpu' if action == 'offload' else 'ram'):
            return False
        self.busy = True
        self.error = ''
        self.state = 'offloading' if action == 'offload' else 'loading'
        self.phase, self.done, self.total = '', 0, 0
        self._revision += 1
        threading.Thread(target=self._request, args=('/' + action, self._revision), daemon=True).start()
        return True

    def poll(self) -> None:
        while True:
            try:
                revision, action, result, error = self._results.get_nowait()
            except queue.Empty:
                break
            if not action:
                self._polling = False
            if revision != self._revision:
                continue
            if action:
                self.busy = False
                self._revision += 1  # Discard polls that preceded completion of this action.
            if result is not None:
                for key in ('state', 'retained_host_bytes', 'phase', 'done', 'total', 'active_requests'):
                    setattr(self, key, result.get(key, 0 if key not in ('state', 'phase') else ''))
            elif action:
                self.state = 'unknown'
                self.error = error
            if action:
                self._next_poll = 0.0
        now = time.monotonic()
        if not self._polling and now >= self._next_poll:
            self._polling = True
            self._next_poll = now + (0.5 if self.busy else 2.0)
            threading.Thread(target=self._request, args=('', self._revision), daemon=True).start()

    @property
    def label(self) -> str:
        name = {'unknown': '等待模型状态', 'gpu': '模型在显存', 'ram': '模型在内存',
                'offloading': '卸载到内存中', 'loading': '加载回显存中'}[self.state]
        if self.retained_host_bytes:
            name += f' · RAM {self.retained_host_bytes / (1024 ** 3):.2f} GiB'
        return name
