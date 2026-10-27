"""Read-only, approximate 27B VRAM planning and observed server load stages.

This is a launcher estimate, not the Engine allocator or an artifact validator.
Only the object directory is read; packed model payloads are never mapped or loaded.
"""

from dataclasses import dataclass
from functools import lru_cache
import csv
import io
import json
import os
from pathlib import Path
import re
import struct
import subprocess

MIB = 1024 ** 2
GIB = 1024 ** 3


@lru_cache(maxsize=8)
def _directory(path: str, size: int, modified: int) -> tuple[dict, ...]:
    with Path(path).open('rb') as stream:
        prefix = stream.read(16)
        if len(prefix) != 16:
            raise ValueError('模型目录不可读')
        magic, length = struct.unpack('<8sQ', prefix)
        if magic not in (b'NINFER\0\2', b'NINFER\0\3'):
            raise ValueError('无法预估此模型格式')
        prefix_bytes = 32 if magic[-1] == 3 else 16
        if not 0 < length <= min(size - prefix_bytes, 32 * MIB):
            raise ValueError('模型目录长度无效')
        if magic[-1] == 3:
            stream.read(16)  # Published v3 framing has a 32-byte prefix.
        directory = json.loads(stream.read(length))
    model = directory.get('identity', {}).get('model_id', directory.get('metadata', {}).get('name'))
    if model not in ('qwen3.6-27b', 'qwen3.8-27b'):
        raise ValueError('当前显存预估适用于已注册的 27B 模型')
    return tuple(dict(obj, name=obj.get('name', obj.get('id'))) for obj in directory['objects'])


@dataclass(frozen=True)
class MemoryOptions:
    tp: int = 2
    pp: int = 1
    context: int = 180000
    capacity: str = ''
    kv_dtype: str = 'int8'
    chunk: int = 2560
    backend: str = 'mtp'
    drafts: int = 4
    optimized: bool = True
    concurrency: int = 1
    vision: bool = False
    vision_device: str = "gpu"
    vision_tokens: int = 1024
    graphs: bool = True
    ram_window: int = 0


@dataclass(frozen=True)
class MemoryEstimate:
    weights: int
    kv: int
    state: int
    workspace_low: int
    workspace_high: int
    graph: int
    auto: bool
    host_vision_bytes: int = 0

    @property
    def low(self) -> int:
        return self.weights + self.kv + self.state + self.workspace_low + self.graph + 256 * MIB

    @property
    def high(self) -> int:
        return self.weights + self.kv + self.state + self.workspace_high + self.graph + 768 * MIB


def estimate_memory(path: Path, options: MemoryOptions) -> list[MemoryEstimate]:
    stat = path.stat()
    objects = _directory(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    o = options
    if o.tp not in (1, 2, 4) or o.pp not in (1, 2) or (o.pp == 2 and o.tp != 1):
        raise ValueError('请选择 TP1/TP2/TP4 或 TP1/PP2')
    if o.context <= 0 or o.chunk <= 0 or o.concurrency <= 0:
        raise ValueError('上下文、chunk 和并发须为正数')
    capacity = o.ram_window or (o.context if o.capacity in ('', 'auto') else int(o.capacity))
    if capacity <= 0:
        raise ValueError('KV 容量须为正数')
    capacity = (capacity + 63) // 64 * 64
    weights = [0] * (o.tp * o.pp)
    host_vision_bytes = 0
    for obj in objects:
        name = obj['name']
        if obj.get('kind') != 'tensor':
            continue
        if name.startswith('vision/'):
            if o.vision and o.vision_device == "gpu":
                weights[0] += obj['bytes']
            elif o.vision and o.vision_device == "cpu":
                elements = 1
                for extent in obj['shape']:
                    elements *= extent
                host_vision_bytes += 4 * elements
            continue
        if name.startswith('mtp/') and o.backend != 'mtp':
            continue
        if name.startswith(('dflash/', 'dflash2/')) and o.backend != 'dflash':
            continue
        if name in ('text/draft_head', 'text/draft_head_token_ids') and not (o.optimized and o.backend != 'none'):
            continue
        if not name.startswith(('text/', 'mtp/', 'dflash/', 'dflash2/')):
            continue
        if o.pp == 2:
            match = re.match(r'text/layers/(\d+)/', name)
            ranks = [1 if int(match[1]) < 32 else 0] if match else [0]
            if name == 'text/token_embedding':
                ranks = [0, 1]
            for rank in ranks:
                weights[rank] += obj['bytes']
        else:
            replicated = (name == 'text/token_embedding' or len(obj.get('shape', [])) < 2 or
                          name.startswith(('dflash/', 'dflash2/')))
            per_rank = obj['bytes'] if replicated else (obj['bytes'] + o.tp - 1) // o.tp
            for rank in range(o.tp):
                weights[rank] += per_rank
    token_bytes = 33792 if o.kv_dtype == 'int8' else 65536
    main_kv = capacity * token_bytes // (o.tp * o.pp)
    # Persistent GDN matrix/conv state, plus a conservative speculative-record allowance.
    gdn = 48 * (48 * 128 * 128 * 4 + 10240 * 3 * 2) // (o.tp * o.pp)
    state = gdn * (2 * o.concurrency + bool(o.ram_window))
    if o.backend != 'none':
        state += 64 * MIB * o.concurrency * (o.drafts + 1) // (5 * o.tp * o.pp)
    chunk = min(o.chunk, o.context)
    # Vocabulary logits and activation working sets dominate measured 27B prefill space.
    workspace = chunk * (248320 * 2 + 5120 * 12) + 32 * MIB
    if o.vision and o.vision_device == "gpu":
        visual = o.vision_tokens * 4 * (1152 * 8 + 4304 * 2) * 2
        workspace = max(workspace, visual + 64 * MIB)
    if o.backend == 'dflash':
        state += (2 * 5 * 2048 * 8 * 128 * 2 * 2 + 5 * 5120 * chunk * 2) * o.concurrency
        workspace += 256 * MIB
    workspace += (o.concurrency - 1) * 32 * MIB
    if o.pp == 2:
        workspace += chunk * 5120 * 2 * 8  # bounded prefill wave buffers
    graph_mib = ((24 if o.context <= 4096 else 94) if o.backend == 'mtp' else
                 (64 if o.context <= 4096 else 96) if o.backend == 'dflash' else
                 20 if o.tp * o.pp > 1 else 12)
    graph = graph_mib * MIB * o.concurrency if o.graphs else 0
    results = []
    for rank, weight in enumerate(weights):
        kv = main_kv
        if o.backend == 'mtp' and (o.pp == 1 or rank == 0):
            kv += (capacity + (64 * o.concurrency if o.drafts > 1 else 0)) * token_bytes // (16 * o.tp)
        transient = o.vision_tokens * 5120 * 2 if o.vision and rank == 0 else 0
        results.append(MemoryEstimate(weight, kv, state, int(workspace * 0.85) + transient,
                                      int(workspace * 1.25) + transient, graph, o.capacity == 'auto' and not o.ram_window,
                                      host_vision_bytes if rank == 0 else 0))
    return results


@dataclass(frozen=True)
class GpuMemory:
    physical: str
    uuid: str
    name: str
    total: int
    used: int
    free: int


def query_gpu_memory() -> dict[int, GpuMemory]:
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
    devices = [GpuMemory(row[0].strip(), row[1].strip(), row[2].strip(),
                         *(int(float(value.strip()) * MIB) for value in row[3:6]))
               for row in csv.reader(io.StringIO(result.stdout)) if len(row) == 6]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        selected = []
        for item in visible.split(','):
            item = item.strip()
            match = next((gpu for gpu in devices if item == gpu.physical or
                          (item.startswith('GPU-') and gpu.uuid.startswith(item))), None)
            if match is None:
                break
            selected.append(match)
        devices = selected
    return dict(enumerate(devices))


class LoadStage:
    def __init__(self):
        self.label = '未启动'
        self.percent: float | None = 0
        self.pending = ''
        self.ready = False

    def feed(self, text: str) -> bool:
        self.pending += text.replace('\r', '\n')
        changed = False
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            match = re.search(r'\bload\s+weights\s+(\d+(?:\.\d+)?)%', line)
            if match and not self.ready:
                self.label = '上传权重'
                self.percent = min(100., float(match[1]))
                if self.percent >= 100:
                    self.label, self.percent = '初始化运行状态', None
                changed = True
            elif 'warming up' in line and not self.ready:
                self.label, self.percent = '预热', None
                changed = True
            elif 'loading model' in line and not self.ready:
                self.label, self.percent = '读取模型目录', None
                changed = True
            elif 'listening on http' in line:
                self.label, self.percent, self.ready = '就绪', 100., True
                changed = True
        return changed
