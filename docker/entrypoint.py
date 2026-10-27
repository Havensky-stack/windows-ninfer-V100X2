"""Pull a native model artifact, then enter the terminal launcher."""
from __future__ import annotations

import argparse
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = 'MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer'
DEFAULT_REVISION = 'v3'
DEFAULT_FILENAME = 'qwen3_8_27b_nvfp4.ninfer'
DEFAULT_MODEL = f'https://huggingface.co/{DEFAULT_REPO}/resolve/{DEFAULT_REVISION}/{DEFAULT_FILENAME}'


def artifact_name(filename: str) -> str:
    path = PurePosixPath(filename)
    if not filename or path.is_absolute() or '..' in path.parts or path.suffix != '.ninfer':
        raise ValueError('Select a .ninfer artifact; raw safetensors/GGUF checkpoints require conversion first')
    return filename


def hub_source(source: str, revision: str | None, filename: str | None) -> tuple[str, str, str | None]:
    """Accept a Hub repo ID, repository URL, tree URL or resolve/blob file URL."""
    if source.startswith(('https://', 'http://')):
        url = urlsplit(source)
        if url.hostname not in ('huggingface.co', 'hf.co'):
            raise ValueError('Expected a Hugging Face URL')
        parts = [unquote(part) for part in url.path.strip('/').split('/')]
        if len(parts) > 2:
            if len(parts) < 4 or parts[2] not in ('tree', 'resolve', 'blob'):
                raise ValueError('Use a Hugging Face repository, tree or file URL')
            revision = revision or parts[3]
            if parts[2] in ('resolve', 'blob'):
                filename = filename or '/'.join(parts[4:])
            elif len(parts) > 4:
                raise ValueError('Use the repository root tree URL and --filename for a subfolder')
        repo_id = '/'.join(parts[:2])
    else:
        repo_id = source
    parts = repo_id.split('/')
    if len(parts) != 2 or any(not part or part in ('.', '..') for part in parts):
        raise ValueError('Expected a Hugging Face owner/repository or model URL')
    if filename is not None:
        artifact_name(filename)
    return repo_id, revision or 'main', filename


def pull_hub(source: str, directory: Path, revision: str | None, filename: str | None) -> Path:
    from huggingface_hub import HfApi, hf_hub_download

    repo_id, revision, filename = hub_source(source, revision, filename)
    if filename is None:
        candidates = sorted(name for name in HfApi().list_repo_files(repo_id, revision=revision)
                            if name.endswith('.ninfer'))
        if len(candidates) != 1:
            raise ValueError('Specify --filename: the repository must contain exactly one .ninfer '
                             f'otherwise. Found: {", ".join(candidates) or "none"}')
        filename = candidates[0]
    artifact_name(filename)
    if repo_id == DEFAULT_REPO and revision == DEFAULT_REVISION:
        destination = directory / 'quasar-v3'
    else:
        key = sha256(f'{repo_id}@{revision}'.encode()).hexdigest()[:16]
        destination = directory / 'hub' / key
    print(f'Pulling {repo_id}@{revision}/{filename}', flush=True)
    # Hub's local metadata reuses complete files and resumes interrupted downloads.
    return Path(hf_hub_download(repo_id=repo_id, revision=revision, filename=filename,
                               local_dir=destination)).resolve()


def pull_url(source: str, directory: Path) -> Path:
    filename = artifact_name(unquote(urlsplit(source).path.rsplit('/', 1)[-1]))
    destination = directory / 'urls' / sha256(source.encode()).hexdigest()[:16] / filename
    if destination.is_file() and destination.stat().st_size:
        return destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.ninfer.partial')
    print(f'Pulling {filename}', flush=True)
    subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--retry-delay', '1',
                    '--continue-at', '-', '--output', str(partial), source], check=True)
    if not partial.stat().st_size:
        raise ValueError('The model download was empty')
    partial.replace(destination)
    return destination.resolve()


def pull_model(source: str, directory: Path, revision: str | None = None,
               filename: str | None = None) -> Path:
    url = urlsplit(source)
    if url.scheme in ('http', 'https'):
        if url.hostname in ('huggingface.co', 'hf.co'):
            return pull_hub(source, directory, revision, filename)
        if revision is not None or filename is not None:
            raise ValueError('--revision and --filename apply only to Hugging Face sources')
        return pull_url(source, directory)
    local = Path(source).expanduser()
    if local.is_file():
        artifact_name(local.name)
        return local.resolve()
    if source.endswith('.ninfer') or source.startswith(('/', '.', '~')):
        raise ValueError(f'Local artifact not found: {source}')
    return pull_hub(source, directory, revision, filename)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog='Other options are forwarded to the TUI, e.g. --profile quasar-long or --print-command.')
    parser.add_argument('--model', default=os.environ.get('NINFER_MODEL') or DEFAULT_MODEL,
                        help='HF repository/file URL, owner/repository, HTTP .ninfer URL or local .ninfer path')
    parser.add_argument('--revision', help='HF revision; URL revision or main by default')
    parser.add_argument('--filename', help='exact .ninfer path in a HF repository; required if ambiguous')
    parser.add_argument('--model-dir', type=Path,
                        default=Path(os.environ.get('NINFER_MODEL_DIR', '/Models/ninfer-V100X2')))
    parser.add_argument('--pull-only', action='store_true', help='download and print the artifact path, then exit')
    args, tui_args = parser.parse_known_args(argv)
    if tui_args[:1] == ['--']:
        tui_args = tui_args[1:]
    if not args.pull_only and '--print-command' not in tui_args:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            parser.error('TUI requires a terminal: use docker run -it or docker compose run; '
                         '--pull-only downloads without a terminal')
    try:
        artifact = pull_model(args.model, args.model_dir, args.revision, args.filename)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Model pull failed: {error}\n')
    print(f'Model: {artifact}', flush=True)
    if args.pull_only:
        return 0
    os.execv(sys.executable, [sys.executable, str(ROOT / 'tools/v100/ninfer_tui.py'),
        '--binary', '/usr/local/bin/ninfer-serve', '--artifact', str(artifact),
        '--host', '0.0.0.0', '--log', '/data/logs/requests.jsonl', *tui_args])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
